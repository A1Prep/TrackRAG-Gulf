"""Read the explicitly released local 8B observations for registered KartaView originals."""
from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

from . import bay_landmarks, local_surfaces
from .parking_evidence import signed_bytes
from .streamed import METADATA_SHA256, STREAMED_MODEL_ID, WEIGHT_SHA256


ORIGIN="local_qwen_release_v1"
BANK=Path(__file__).resolve().parents[2]/"experiments/dropoff/passenger-local-8b-observations-v1.json"
BANK_SHA="d570c39941902351fd61ea194bb567b4372612cffcb80c3cd8226c8e1b985c99"


@lru_cache(maxsize=1)
def _database():
    try:
        data=BANK.read_bytes().replace(b"\r\n",b"\n")
        if sha256(data).hexdigest()!=BANK_SHA:
            return {}
        record=json.loads(data)
        if record.get("origin")!=ORIGIN or record.get("version")!=1:
            return {}
        return record["sources"]
    except (KeyError,OSError,TypeError,ValueError):
        return {}


def get_packet(key, kind):
    if kind not in {"points","landmarks"}:
        raise ValueError("Unknown local visual observation type.")
    packet=_database().get(key,{}).get(kind)
    return deepcopy(packet)


def verify(packet, key, kind, *, scene=None):
    official=_database().get(key,{}).get(kind)
    if not isinstance(official,dict) or not isinstance(packet,dict):
        raise ValueError("This original has no released local visual observations.")
    body=packet.get("body",{})
    if (set(packet)!={"origin","body_sha256","body"} or packet["origin"]!=ORIGIN
            or packet["body_sha256"]!=official["body_sha256"]
            or sha256(signed_bytes(body)).hexdigest()!=official["body_sha256"]):
        raise ValueError("The local visual observation differs from the recorded model response.")
    size=body["size"]
    registration={"source":"predicted","provenance":{"image_sha256":key,"network_policy":"local_only"},
                  "frame":{"recording_id":body["recording_id"],"frame_id":body["frame_id"],"width":size[0],"height":size[1]}}
    if (body["image_sha256"]!=key or not local_surfaces.matches_scene(registration) or scene is not None
            and (scene.get("provenance",{}).get("network_policy")!="local_only" or
                 scene.get("provenance",{}).get("image_sha256")!=key or not local_surfaces.matches_scene(scene))):
        raise ValueError("Local visual observations require the registered current original.")
    model=body["local_model"]
    weights=model["weights"]
    expected={**WEIGHT_SHA256,**METADATA_SHA256}
    if (model["base"]!=bay_landmarks.MODELS["base"] or model["processor"]!=bay_landmarks.MODELS["processor"]
            or model["adapter"]!=bay_landmarks.MODELS["adapter"] or model["runtime"]!="local_streamed"
            or model["quantization"]!="nf4" or weights["model_id"]!=STREAMED_MODEL_ID
            or weights["revision"]!=model["base"] or weights["verified_weights"] is not True
            or weights["verified_metadata"] is not True or {k:v["sha256"] for k,v in weights["files"].items()}!=expected):
        raise ValueError("The neutral observations came from a different checkpoint or adapter.")
