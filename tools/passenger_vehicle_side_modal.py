"""Run one source-allowlisted, neutral side-ground observation batch on the cached 35B reader."""
import base64
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import tempfile

import modal

from tools import passenger_qwen_point_protocol as protocol
from tools.passenger_qwen35_download import MODEL_FOLDER, REPO, REVISION, VOLUME
from tools.passenger_roadside35 import image


app = modal.App("trackrag-public-vehicle-side-ground")
if modal.is_local():
    root = Path(__file__).resolve().parents[1]
    image = image.add_local_file(root/"trackrag/dropoff/vehicle_side_view.py", "/root/trackrag/dropoff/vehicle_side_view.py")


@app.function(image=image, gpu="H100", cpu=4, memory=55000, timeout=850, max_containers=1,
              single_use_containers=True, retries=0,
              volumes={"/weights": modal.Volume.from_name(VOLUME).read_only()},
              secrets=[modal.Secret.from_name("trackrag-parking-receipt-v1")])
def observe(jobs):
    from PIL import Image
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from trackrag.dropoff import vehicle_side_view as view

    allowed = set(protocol.read_allowlist("/trial-public").values())
    counts = [row["role"] for job in jobs for row in job["rows"]]
    if (not 0 < len(jobs) <= 16 or counts.count("candidate") != 31 or len(counts) > 79
            or set(counts) - {"candidate", "wrong_side_control", "occupied_control", "lane_control"}):
        raise ValueError("Batch exceeds the declared public side-ground trial")
    checkpoint = Path("/weights")/MODEL_FOLDER
    proof = json.loads((checkpoint/"trial-download-verification.json").read_bytes())
    if (proof["repository"], proof["revision"], proof["shards"], proof["shard_bytes"]) != (REPO, REVISION, 42, 37463662160):
        raise ValueError("Verified official model volume differs from the requested snapshot")
    signer = Ed25519PrivateKey.from_private_bytes(base64.b64decode(os.environ.pop("PARKING_SIGNING_KEY"), validate=True))
    boards, bodies = [], {}
    for job in jobs:
        key = sha256(job["photo"]).hexdigest()
        if key != job["source"] or key not in allowed or key in bodies:
            raise ValueError("Unexpected original photograph")
        records = []
        with Image.open(BytesIO(job["photo"])) as image:
            for row in job["rows"]:
                calls = []
                for close in (False, True):
                    board = view.render(image, row, close=close)
                    boards.append(base64.b64encode(board).decode("ascii"))
                    calls.append({"image": sha256(board).hexdigest()})
                records.append({"spec": view.digest(row), "calls": calls})
        bodies[key] = {"recipe": view.RECIPE, "source": key, "context": job["context"], "model": REPO,
                       "revision": REVISION, "guide": view.digest(view.GUIDE), "rows": records}
    print(f"Prepared {len(boards)} wide/close photographs from {len(jobs)} approved originals", flush=True)
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        data, out, prompt = work/"images.json", work/"answers.json", work/"prompt.txt"
        data.write_text(json.dumps(boards))
        prompt.write_text(view.GUIDE)
        subprocess.run(["/usr/bin/python3.12", "-u", "/root/tools/passenger_roadside35_native.py",
                        str(data), str(out), str(checkpoint), str(prompt)], check=True, timeout=725)
        replies = json.loads(out.read_bytes())
    if len(boards) != len(replies):
        raise ValueError("Incomplete neutral visual batch")
    replies = iter(replies)
    packets = {}
    for key, body in bodies.items():
        for row in body["rows"]:
            for observation in row["calls"]:
                observation["verdict"] = view.verdict(next(replies))
        payload = json.dumps(body, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
        packets[key] = {"body": body, "signature": signer.sign(payload).hex()}
    return packets


@app.local_entrypoint()
def main():
    from tools.passenger_release import ROOT, PUBLIC_PHOTO_LISTS
    folder = ROOT/"runs/passenger-placement-field-20261124/vehicle-side"
    gallery = ROOT/"data/uae-parking-study"
    allowed = {p["sha256"] for name in PUBLIC_PHOTO_LISTS
               for p in json.loads((ROOT/"experiments/dropoff"/name).read_bytes())["photos"]}
    sources = json.loads((folder/"jobs.json").read_bytes())
    jobs = []
    for saved in sources:
        key = saved["source"]
        if key not in allowed:
            raise ValueError("Source is outside the authorized public gallery")
        photo = (gallery/(key+".jpg")).read_bytes()
        if sha256(photo).hexdigest() != key:
            raise ValueError("Public original changed")
        jobs.append({**saved, "photo": photo})
    print(f"Starting one neutral {len(jobs)}-photo batch on the existing verified checkpoint", flush=True)
    packets = observe.remote(jobs)
    temp = folder/"responses.pending"
    temp.write_text(json.dumps(packets, separators=(",", ":")), encoding="utf-8")
    temp.replace(folder/"responses.json")
    print(f"Downloaded {len(packets)} signed observations", flush=True)
