"""Verify locally prepared maps against the versioned source and model output manifest."""
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

from tools.passenger_parking_study import SEG_WEIGHTS
from .bay_packets import read_photo_packet
from .parking_evidence import signed_bytes


BANK = Path(__file__).resolve().parents[2] / "experiments/dropoff/passenger-local-image-surfaces-v1.zip"
INDEX_SHA256 = "ecf4a15de3b2b1c92682ef50196f9e1352d2de958fc7c9eed845240506708b1c"


@lru_cache(maxsize=1)
def _sources():
    try:
        raw = BANK.with_suffix(".json").read_bytes().replace(b"\r\n",b"\n")
        if sha256(raw).hexdigest() != INDEX_SHA256:
            return {}
        manifest = json.loads(raw)
        if manifest["version"] == 1 and manifest["weights_sha256"] == SEG_WEIGHTS:
            return manifest["sources"]
    except (KeyError, ValueError, OSError):
        pass
    return {}


def local_only(scene):
    provenance = scene.get("provenance", {})
    return provenance.get("network_policy") == "local_only" or provenance.get("image_sha256") in _sources()


def matches_scene(scene):
    row = _sources().get(scene.get("provenance", {}).get("image_sha256"))
    if not row:
        return False
    frame = scene["frame"]
    return bool(scene.get("source") == "predicted" and
                (row["recording_id"], row["frame_id"], row["size"]) ==
                (frame["recording_id"], frame["frame_id"], [frame["width"], frame["height"]]))


def verify(packet, frame, photo_sha):
    row = _sources().get(photo_sha)
    body = packet.get("body")
    if not row or not isinstance(body,dict):
        raise ValueError("This local surface has no released source record.")
    expected = (row["recording_id"], row["frame_id"], row["size"], SEG_WEIGHTS, photo_sha)
    actual = tuple(body.get(k) for k in ("recording_id","frame_id","size","weights_sha256","image_sha256"))
    if (actual != expected or [frame.get("width"),frame.get("height")] != row["size"]
            or row["body_sha256"] != sha256(signed_bytes(body)).hexdigest()
            or packet.get("signature") != row["body_sha256"]
            or packet.get("origin") != "local_release_v1"
            or frame.get("recording_id", row["recording_id"]) != row["recording_id"]
            or frame.get("frame_id", row["frame_id"]) != row["frame_id"]):
        raise ValueError("Local surface differs from the recorded inference or original frame.")


def get_packet(photo_sha):
    if photo_sha not in _sources():
        return None
    return read_photo_packet(BANK,photo_sha,1_500_000)
