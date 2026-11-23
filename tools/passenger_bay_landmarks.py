"""Prepare public, source-bound storefront observations from the project's trained Qwen3-VL 8B."""
import base64
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import modal
from tools.passenger_place_modal import TrainedPlaceReader, image

app = modal.App("trackrag-painted-bay-landmark-study")
if modal.is_local():
    from tools.passenger_release import PUBLIC_PHOTO_LISTS, ROOT
    image = image.add_local_file(ROOT/"tools/passenger_place_modal.py", "/root/tools/passenger_place_modal.py")
    for source in PUBLIC_PHOTO_LISTS:
        image = image.add_local_file(ROOT/"experiments/dropoff"/source, "/sources/"+source)


@app.cls(image=image, gpu="A100-40GB", cpu=4, memory=24000, timeout=220, startup_timeout=300,
         min_containers=0, max_containers=1, scaledown_window=15, retries=0,
         volumes={"/base":modal.Volume.from_name("trackrag-qwen8b-public-weights").read_only()},
         secrets=[modal.Secret.from_name("trackrag-parking-receipt-v1")])
class PublicLandmarks(TrainedPlaceReader):
    @modal.enter()
    def load(self):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        self.key = Ed25519PrivateKey.from_private_bytes(base64.b64decode(os.environ.pop("PARKING_SIGNING_KEY"),validate=True))
        self.known = {p["sha256"] for f in Path("/sources").glob("*.json") for p in json.loads(f.read_bytes())["photos"]}
        self.qwen = None

    @modal.method()
    def describe(self, photos):
        from PIL import Image
        from trackrag.dropoff import bay_landmarks as landmarks
        from trackrag.dropoff.parking_evidence import MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS, signed_bytes
        if not isinstance(photos,list) or not 1 <= len(photos) <= 4:
            raise ValueError("Expected one to four published photographs.")
        packets = []
        for photo in photos:
            if not isinstance(photo,bytes) or not 0 < len(photo) <= MAX_IMAGE_BYTES:
                raise ValueError("Invalid public photo.")
            key = sha256(photo).hexdigest()
            if key not in self.known:
                raise ValueError("Only published originals in the release can be described.")
            with Image.open(BytesIO(photo)) as original:
                if original.width*original.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Photo exceeds model dimensions.")
                size = list(original.size)
                bands = []
                for i in range(len(landmarks.BANDS)):
                    pixels = landmarks.band_image(original,i)
                    words = self.read(landmarks.GUIDE, landmarks.WORDS, pixels)
                    bands.append({"view_sha256":sha256(pixels).hexdigest(), "places":landmarks.public_phrase(words)})
            body = {"recipe":landmarks.RECIPE,"models":landmarks.MODELS,"image_sha256":key,"size":size,
                    "prompt_sha256":sha256((landmarks.GUIDE+"\n"+landmarks.WORDS).encode()).hexdigest(),"bands":bands}
            packets.append({"body":body,"signature":self.key.sign(signed_bytes(body)).hex()})
        return packets


@app.local_entrypoint()
def main():
    from trackrag.dropoff import bay_landmarks as landmarks
    from trackrag.dropoff.parking_evidence import signed_bytes
    from tools.passenger_release import PUBLIC_PHOTO_LISTS, ROOT
    keys = {p["sha256"] for name in PUBLIC_PHOTO_LISTS
            for p in json.loads((ROOT/"experiments/dropoff"/name).read_bytes())["photos"]}
    cache = ROOT/"runs/model-cache/bay-landmarks-v1"
    cache.mkdir(parents=True,exist_ok=True)
    originals = ROOT/"data/uae-parking-study"
    pending, known = [], {}
    for key in sorted(keys):
        path = originals/(key+".jpg")
        if sha256(path.read_bytes()).hexdigest() != key:
            raise ValueError("Public photo differs from the manifest.")
        saved = cache/(key+".json")
        if saved.is_file():
            packet = json.loads(saved.read_bytes())
            landmarks.verify(packet,path.read_bytes(),packet["body"]["size"])
            known[key] = packet
        else:
            pending.append(key)
    worker = PublicLandmarks()
    for offset in range(0,len(pending),4):
        batch = pending[offset:offset+4]
        photos = [(originals/(key+".jpg")).read_bytes() for key in batch]
        for key,photo,packet in zip(batch, photos, worker.describe.remote(photos), strict=True):
            landmarks.verify(packet, photo, packet["body"]["size"])
            saved = cache/(key+".json")
            temporary = saved.with_suffix(".tmp")
            temporary.write_bytes(signed_bytes(packet))
            os.replace(temporary,saved)
            known[key] = packet
        print(f"Verified public landmark records for {len(known)}/{len(keys)} originals.",flush=True)
    partial = landmarks.BANK.with_suffix(".zip.tmp")
    with ZipFile(partial,"w",ZIP_DEFLATED) as archive:
        for key in sorted(keys):
            archive.writestr(key+".json",signed_bytes(known[key]))
    os.replace(partial,landmarks.BANK)
