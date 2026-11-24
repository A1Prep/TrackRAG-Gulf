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
_extra_sources={}


def _read_bank(path, expected_sha):
    data=Path(path).read_bytes().replace(b"\r\n",b"\n")
    if sha256(data).hexdigest()!=expected_sha.lower():
        raise ValueError("The 8B observation file differs from its supplied SHA-256.")
    record=json.loads(data)
    if record.get("origin")!=ORIGIN or record.get("version")!=1 or not isinstance(record.get("sources"),dict):
        raise ValueError("Unsupported local 8B observations.")
    return record["sources"]


@lru_cache(maxsize=1)
def _database():
    try:
        rows=_read_bank(BANK,BANK_SHA)
    except (KeyError,OSError,TypeError,ValueError):
        rows={}
    return {**rows, **_extra_sources}


def register(path, expected_sha):
    rows=_read_bank(path,expected_sha)
    if not rows or _database().keys() & rows.keys() or rows.keys()-local_surfaces._sources().keys():
        raise ValueError("Additional 8B observations require distinct registered original surfaces.")
    for key,entry in rows.items():
        if set(entry)!={"points","landmarks"}:
            raise ValueError("Local observations require both the ground checks and the landmark views.")
        for packet in entry.values():
            _verify(packet,key,packet)
    _extra_sources.update(rows)
    _database.cache_clear()


def get_packet(key, kind):
    if kind not in {"points","landmarks"}:
        raise ValueError("Unknown local visual observation type.")
    packet=_database().get(key,{}).get(kind)
    return deepcopy(packet)


def verify(packet, key, kind, *, scene=None):
    official=_database().get(key,{}).get(kind)
    _verify(packet,key,official,scene=scene)


def _verify(packet,key,official,*,scene=None):
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
