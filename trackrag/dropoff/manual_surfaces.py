"""Read verified semantic parking maps without sending passenger words or marks to a service."""
import json
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile, ZipFile

from .parking_evidence import MAX_IMAGE_PIXELS, _authenticate, _decode_map, signed_bytes

RECIPE = "public_manual_parking_surface_v1"
REVISION = "4772b6bf101d91f2534c106dc524d906aeb3c68a"
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT/"runs/model-cache/manual-parking-surfaces-v1"
BANK = ROOT/"experiments/dropoff/passenger-manual-parking-surfaces-20261123.zip"
MAX_PACKET = 1_500_000


def decode(packet, frame, photo_sha):
    _authenticate(packet)
    body=packet["body"]
    size=[frame["width"],frame["height"]]
    if (body.get("recipe")!=RECIPE or body.get("model_revision")!=REVISION
            or not isinstance(photo_sha,str) or not re.fullmatch(r"[a-f0-9]{64}",photo_sha)
            or body.get("image_sha256")!=photo_sha or body.get("size")!=size
            or any(type(v) is not int or v<=0 for v in size) or size[0]*size[1]>MAX_IMAGE_PIXELS):
        raise ValueError("Parking evidence differs from the original photo, resolution or model.")
    return _decode_map(body["class_map"],frame)


def get_packet(photo_sha):
    if not isinstance(photo_sha,str) or not re.fullmatch(r"[a-f0-9]{64}",photo_sha):
        return None
    name=photo_sha+".json"
    path=CACHE/name
    try:
        if path.stat().st_size<=MAX_PACKET:
            packet = json.loads(path.read_bytes())
            width, height = packet["body"]["size"]
            decode(packet, {"width":width, "height":height}, photo_sha)
            return packet
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        pass
    try:
        with ZipFile(BANK) as bank:
            if bank.getinfo(name).file_size<=MAX_PACKET:
                return json.loads(bank.read(name))
    except (BadZipFile,OSError,ValueError,KeyError):
        pass
    return None


def save(packet):
    body=packet["body"]
    width,height=body["size"]
    decode(packet,{"width":width,"height":height},body["image_sha256"])
    CACHE.mkdir(parents=True,exist_ok=True)
    target=CACHE/(body["image_sha256"]+".json")
    with NamedTemporaryFile(dir=CACHE,delete=False) as tmp:
        temporary=Path(tmp.name)
        tmp.write(signed_bytes(packet))
    try:
        os.replace(temporary,target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
