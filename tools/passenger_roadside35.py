"""Build an original-bound public photo index from the frozen 35B visual model on Modal."""
import base64
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED,ZipFile

import modal

from tools import passenger_qwen_point_protocol as protocol
from tools.passenger_qwen35_download import MODEL_FOLDER,REPO,REVISION,VOLUME
from tools.passenger_qwen35_trial import IMAGE

app=modal.App("trackrag-public-roadside-context")
image=(modal.Image.from_registry(IMAGE,add_python="3.11").entrypoint([])
       .pip_install("Pillow==11.1.0","cryptography==50.0.2"))
if modal.is_local():
    root=Path(__file__).resolve().parents[1]
    image=protocol.mount_trial(image)
    image=image.add_local_file(root/"trackrag/dropoff/roadside_board.py","/root/trackrag/dropoff/roadside_board.py")
    image=image.add_local_file(root/"tools/passenger_roadside35_native.py","/root/tools/passenger_roadside35_native.py")


@app.function(image=image,gpu="H100",cpu=6,memory=56000,timeout=1500,retries=0,max_containers=1,single_use_containers=True,
              secrets=[modal.Secret.from_name("trackrag-parking-receipt-v1")],
              volumes={"/weights":modal.Volume.from_name(VOLUME).read_only()})
def inspect(rows):
    import re
    import subprocess
    import tempfile
    from PIL import Image
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from trackrag.dropoff import roadside_board as board
    def canonical(data):
        return json.dumps(data,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("utf-8")
    def digest(data):
        return sha256(canonical(data)).hexdigest()
    known=set(protocol.read_allowlist("/trial-public").values())
    signer=Ed25519PrivateKey.from_private_bytes(base64.b64decode(os.environ.pop("PARKING_SIGNING_KEY"),validate=True))
    if not rows or len(rows)>len(known) or sum(len(r["specs"]) for r in rows)>550:
        raise ValueError("Unexpected public observation batch.")
    folder=Path("/weights")/MODEL_FOLDER
    receipt=json.loads((folder/"trial-download-verification.json").read_bytes())
    if (receipt["repository"],receipt["revision"],receipt["shards"])!=(REPO,REVISION,42):
        raise ValueError("The 35B checkpoint differs from the verified download.")
    jobs=[]; packets={}
    for item in rows:
        key=sha256(item["photo"]).hexdigest()
        if key not in known or key in packets:
            raise ValueError("Unrecognized or duplicate public photograph.")
        with Image.open(BytesIO(item["photo"])) as original:
            observations=[]
            for spec in item["specs"]:
                view=board.render(original,spec)
                observations.append({"spec":digest(spec),"view":sha256(view).hexdigest()})
                jobs.append(base64.b64encode(view).decode("ascii"))
        packets[key]={"recipe":board.RECIPE,"model":REPO,"revision":REVISION,"prompt":digest(board.GUIDE),
                      "image_sha256":key,"context":item["context"],"rows":observations}
    with tempfile.TemporaryDirectory() as temp:
        work=Path(temp); inputs=work/"inputs.json"; out=work/"replies.json"; prompt=work/"guide"
        inputs.write_text(json.dumps(jobs)); prompt.write_text(board.GUIDE)
        subprocess.run(["/usr/bin/python3.12","-u","/root/tools/passenger_roadside35_native.py",
                        str(inputs),str(out),str(folder),str(prompt)],check=True,timeout=1380)
        replies=iter(json.loads(out.read_bytes()))
    results={}
    fallback={k:"NONE" for k in ("nearer_vehicle","farther_vehicle","frontage","roadside")}
    for key,body in packets.items():
        for obs in body["rows"]:
            raw=next(replies).strip()
            try:
                parsed=json.loads(raw[raw.index("{"):raw.rindex("}")+1])
                status=str(parsed["status"]).strip().upper()
                if status not in {"OPEN","LANE","BLOCKED","SEPARATED","UNCLEAR"}: raise ValueError
                desc={k:re.sub(r"[\w.+-]+@[\w.-]+|(?:\+?\d[\s()+-]?){5,}","",str(parsed[k]))[:240] for k in fallback}
                obs["description"]={**desc,"status":status}
            except (ValueError,KeyError,TypeError):
                obs["description"]={**fallback,"status":"UNCLEAR"}
        results[key]={"body":body,"signature":signer.sign(canonical(body)).hex()}
    return results


@app.local_entrypoint()
def main(gravel_only: bool = False):
    from trackrag.dropoff import curb_points,manual_surfaces,roadside_context as c,surface_stalls
    from trackrag.dropoff.passenger import digest
    from tools.passenger_release import PUBLIC_PHOTO_LISTS,ROOT
    bank=ROOT/"data/uae-parking-study"
    allowed={p["sha256"] for f in PUBLIC_PHOTO_LISTS for p in json.loads((ROOT/"experiments/dropoff"/f).read_bytes())["photos"]}
    jobs=[]; seen=set(); inputs={}
    for line in (bank/"passenger-requests.jsonl").read_bytes().splitlines():
        request=json.loads(line); scene=request["scene"]; key=scene["provenance"]["image_sha256"]
        if key in seen or key not in allowed: continue
        seen.add(key); photo=(bank/(key+".jpg")).read_bytes()
        if sha256(photo).hexdigest()!=key: raise ValueError("Source changed.")
        surface=manual_surfaces.get_packet(key)
        if not surface: continue
        pairs=c.candidates(request,photo,surface,surface_stalls.get_packet(key),gravel_only=gravel_only)
        if not pairs: continue
        inputs[key]=(scene,photo,pairs)
        specs=json.loads(json.dumps([spec for spec,_ in pairs],allow_nan=False))
        jobs.append({"photo":photo,"context":digest(curb_points.source_context(scene)),"specs":specs})
    print(f"Inspecting {sum(len(j['specs']) for j in jobs)} measured locations in {len(jobs)} originals from {len(seen)}.",flush=True)
    destination = c.GRAVEL_BANK if gravel_only else c.BANK
    results = inspect.remote(jobs) if jobs else {}
    if set(results)!=set(inputs): raise ValueError("Incomplete result bank.")
    cache = ROOT / "runs" / "model-cache" / destination.stem
    cache.mkdir(parents=True, exist_ok=True)
    for key, packet in results.items():
        (cache / (key + ".json")).write_text(json.dumps(packet, ensure_ascii=False), encoding="utf-8")
    target=destination.with_suffix(".tmp")
    with ZipFile(target,"w",ZIP_DEFLATED) as archive:
        for key in sorted(results):
            c.observations(*inputs[key],results[key])
            archive.writestr(key+".json",json.dumps(results[key],ensure_ascii=False).encode("utf-8"))
    os.replace(target,destination)
    print(f"Verified and saved {len(results)} original-photo observation packets.",flush=True)
