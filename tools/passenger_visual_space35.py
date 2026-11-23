"""Build the two-view road-photo observations with the existing private checkpoint on Modal."""
import base64
from hashlib import sha256
import json
import os
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

import modal
from tools import passenger_qwen_point_protocol as protocol
from tools.passenger_qwen35_download import MODEL_FOLDER,REPO,REVISION,VOLUME
from tools.passenger_roadside35 import image


app=modal.App("trackrag-public-parking-views")
if modal.is_local():
    root=Path(__file__).resolve().parents[1]
    for file in ("trackrag/dropoff/visual_space_board.py","tools/passenger_visual_space_native.py"):
        image=image.add_local_file(root/file,"/root/"+file)


@app.function(image=image,gpu="H100",cpu=6,memory=56000,max_containers=1,timeout=1800,retries=0,single_use_containers=True,
              volumes={"/weights":modal.Volume.from_name(VOLUME).read_only()},
              secrets=[modal.Secret.from_name("trackrag-parking-receipt-v1")])
def inspect(requests):
    import subprocess
    import tempfile
    from io import BytesIO
    from PIL import Image
    from trackrag.dropoff import visual_space_board as board
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    known=set(protocol.read_allowlist("/trial-public").values())
    if not 0<len(requests)<=len(known):
        raise ValueError("Unexpected public photo set.")
    signer=Ed25519PrivateKey.from_private_bytes(base64.b64decode(os.environ.pop("PARKING_SIGNING_KEY"),validate=True))
    weights=Path("/weights")/MODEL_FOLDER
    manifest=json.loads((weights/"trial-download-verification.json").read_bytes())
    if (manifest["repository"],manifest["revision"],manifest["shards"])!=(REPO,REVISION,42):
        raise ValueError("Model weights differ from the verified release.")
    with tempfile.TemporaryDirectory() as temp:
        work=Path(temp)
        descriptions=[]
        seen=set()
        for row in requests:
            photo=row["photo"]
            key=sha256(photo).hexdigest()
            if key not in known or key in seen or any(len(rows)>20 for rows in row["sides"].values()):
                raise ValueError("Unsupported source photograph or candidate set.")
            seen.add(key)
            with Image.open(BytesIO(photo)) as original:
                for side,samples in row["sides"].items():
                    if side not in {"left","right"}:
                        raise ValueError("Unexpected side of original.")
                    (work/f"{key}.{side}.jpg").write_bytes(board.selection(original,samples,side))
                    for i,sample in enumerate(samples,1):
                        (work/f"{key}.{side}.{i}.jpg").write_bytes(board.inspection(original,sample))
            descriptions.append({"source":key, "context":row["context"], "sides":row["sides"]})
        (work/"views.json").write_text(json.dumps(descriptions))
        subprocess.run(["/usr/bin/python3.12","-u","-m","tools.passenger_visual_space_native",
                        str(work),str(weights),REVISION],cwd="/root",check=True,timeout=1700)
        bodies=json.loads((work/"packets.json").read_bytes())
    return {key:{"body":body,"signature":signer.sign(json.dumps(body,separators=(",",":"),sort_keys=True,
                                                                 ensure_ascii=True,allow_nan=False).encode()).hex()}
            for key,body in bodies.items()}


@app.local_entrypoint()
def main(reuse:bool=False, paved:bool=False):
    from io import BytesIO
    from PIL import Image
    from trackrag.dropoff import manual_surfaces,visual_space
    from trackrag.dropoff.passenger import digest
    from tools.passenger_release import PUBLIC_PHOTO_LISTS,ROOT
    bank=ROOT/"data/uae-parking-study"
    allowed={p["sha256"] for f in PUBLIC_PHOTO_LISTS for p in json.loads((ROOT/"experiments/dropoff"/f).read_bytes())["photos"]}
    sources={}; jobs=[]
    for line in (bank/"passenger-requests.jsonl").read_bytes().splitlines():
        scene=json.loads(line)["scene"]
        key=scene["provenance"]["image_sha256"]
        if key not in allowed or key in sources:
            continue
        surface=manual_surfaces.get_packet(key)
        if not surface:
            continue
        photo=(bank/(key+".jpg")).read_bytes()
        if sha256(photo).hexdigest()!=key:
            raise ValueError("Original file changed.")
        classes=manual_surfaces.decode(surface,scene["frame"],key)
        with Image.open(BytesIO(photo)) as original:
            points=visual_space.paved_samples(scene,classes,original) if paved else visual_space.samples(scene,classes)
        if points:
            sources[key]=(scene,photo,surface)
            jobs.append({"photo":photo,"sides":points,"context":digest(visual_space.context(scene))})
    destination=visual_space.PAVED_BANK if paved else visual_space.BANK
    cache=ROOT/"runs/model-cache"/destination.stem
    cache.mkdir(parents=True,exist_ok=True)
    print(f"Reviewing {len(jobs)} allowlisted originals, {sum(len(j['sides']) for j in jobs)} photo sides.",flush=True)
    if reuse:
        packets={key:json.loads((cache/(key+".json")).read_bytes()) for key in sources}
    else:
        packets=inspect.remote(jobs)
        for key,packet in packets.items():
            (cache/(key+".json")).write_text(json.dumps(packet),encoding="utf8")
    if set(packets)!=set(sources):
        raise ValueError("Incomplete photo observations.")
    pending=destination.with_suffix(".tmp")
    count=0
    with ZipFile(pending,"w",ZIP_DEFLATED) as archive:
        for key in sorted(packets):
            count+=len(visual_space.verified(*sources[key],packets[key],paved=paved))
            archive.writestr(key+".json",json.dumps(packets[key]).encode())
    pending.replace(destination)
    print(f"Verified {len(packets)} original photo receipts with {count} visually agreed points before physical checks.",flush=True)
