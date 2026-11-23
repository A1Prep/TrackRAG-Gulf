"""Bind a real GPU decision to its frame and replay the complete decision locally before confirmation."""
import base64
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from tools.passenger_parking_study import SEG_REVISION, QWEN_REVISION
from .contracts import validate_request
from .data import inference_input_sha256
from .passenger import digest
from .passenger_parking import METHOD, parking_instruction
from .parking_steps import steps


RECIPE = "current-vehicle-pair-supported-parking-1"
REVIEWED_RECIPE = "current-vehicle-pair-supported-parking-2"
INSPECTED_RECIPE = "current-vehicle-pair-supported-parking-3"
CALL_LIMITS = {RECIPE: 4, REVIEWED_RECIPE: 5, INSPECTED_RECIPE: 5}
PUBLIC_KEY = "03e686ae61db1b9c6e1b25d8b7efa7d9b42d8628d3e1abfaf7e247db69204607"
MAX_IMAGE_BYTES = 40 * 1024 * 1024
MAX_IMAGE_PIXELS = 42_000_000


def signed_bytes(body):
    return json.dumps(body, ensure_ascii=True, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf8")


def source_input(request, photo):
    request = validate_request(request)
    scene = request["scene"]
    frame = scene["frame"]
    if scene["source"] != "predicted":
        raise ValueError("Live passenger grounding requires predicted observations")
    image_sha = sha256(photo).hexdigest()
    if image_sha != scene.get("provenance", {}).get("image_sha256"):
        raise ValueError("Source photograph no longer matches its observations")
    if len(photo) > MAX_IMAGE_BYTES or frame["width"] * frame["height"] > MAX_IMAGE_PIXELS:
        raise ValueError("Photograph exceeds the supported input size")
    with Image.open(BytesIO(photo)) as image:
        if image.size != (frame["width"], frame["height"]):
            raise ValueError("Source photograph dimensions changed")
    return request, image_sha, inference_input_sha256(request, image_sha)


def current_only(request):
    clean = validate_request(request)
    scene = clean["scene"]
    scene["tracks"] = [t for t in scene["tracks"] if t["frame_id"] == scene["frame"]["frame_id"]]
    return clean


def map_shape(frame):
    w, h = frame["width"], frame["height"]
    factor = min(1., 900 / max(w, h))
    return max(1, round(h * factor)), max(1, round(w * factor))


def _decode_map(record, frame):
    encoded = record["png"]
    if not isinstance(encoded, str) or len(encoded) > 2_000_000:
        raise ValueError("Invalid parking-map payload")
    data = base64.b64decode(encoded, validate=True)
    if sha256(data).hexdigest() != record["sha256"]:
        raise ValueError("Parking map has changed")
    h, w = map_shape(frame)
    with Image.open(BytesIO(data)) as image:
        if image.format != "PNG" or image.mode != "L" or image.size != (w, h):
            raise ValueError("Unexpected parking-map format or size")
        classes = np.array(image)
    if int(classes.max()) > 64:
        raise ValueError("Unknown parking surface class")
    return classes


def view_digest(view):
    """Bind the displayed geometry without depending on private numeric diagnostics or local paths."""
    def visible(option):
        return {k:option[k] for k in ("choice","point","region_box_xyxy","anchor_boxes","tracks","surface","requires_visual_review")}
    if view["type"] == "points":
        return digest({"type":"points","options":[visible(o) for o in view["options"]]})
    if view["type"] == "parking_components":
        return digest({"type":"parking_components",
                       "options":[{k:o[k] for k in ("choice","point","anchor_boxes","tracks")} for o in view["options"]]})
    if view["type"] == "inspection":
        return digest({"type":"inspection","option":visible(view["option"])})
    return digest(view)


def replay_decision(request, classes, calls, photo, recipe=RECIPE):
    if recipe not in tuple(CALL_LIMITS):
        raise ValueError("Unknown parking decision recipe")
    if not isinstance(calls, list) or len(calls) > CALL_LIMITS[recipe]:
        raise ValueError("Invalid number of parking model responses")
    with TemporaryDirectory(prefix="parking-verify-") as folder:
        original = Path(folder)/"original.frame"
        original.write_bytes(photo)
        return _replay(request, classes, calls, original, recipe)


def _replay(request, classes, calls, original, recipe):
    from .parking_views import render
    from .parking_reference_board import steps as reviewed_steps
    from .parking_gap_review import steps as inspected_steps
    algorithm = {RECIPE: steps, REVIEWED_RECIPE: reviewed_steps, INSPECTED_RECIPE: inspected_steps}[recipe]
    machine = algorithm(current_only(portable_request(request)), classes)
    index = 0
    try:
        step = next(machine)
        while index < len(calls):
            call = calls[index]
            if (call.get("kind") != step["kind"] or call.get("view_sha256") != view_digest(step["view"])
                    or call.get("system_sha256") != sha256(step["system"].encode()).hexdigest()
                    or not re.fullmatch(r"[0-9a-f]{64}", str(call.get("rendered_sha256", "")))
                    or not isinstance(call.get("response"), str) or len(call["response"]) > 16384):
                raise ValueError("A model response differs from the recorded input or decision order")
            if step["kind"] == "anchors" and call["rendered_sha256"] != request["scene"]["provenance"]["image_sha256"]:
                raise ValueError("The anchor model did not see the original photograph")
            if step["view"]["type"] != "original":
                marked = render(original, request["scene"]["frame"], step["view"], original.with_suffix(".jpg"))
                if sha256(marked.read_bytes()).hexdigest() != call["rendered_sha256"]:
                    raise ValueError("Model-visible parking image differs on local replay; use the release image libraries")
            index += 1
            step = machine.send(call["response"])
    except StopIteration as done:
        if index != len(calls):
            raise ValueError("Extra model responses after the final decision")
        return done.value
    raise ValueError("Incomplete parking decision")


def _authenticate(packet):
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    if not isinstance(packet, dict) or set(packet) != {"body", "signature"} or not isinstance(packet["body"], dict):
        raise ValueError("A signed parking-model response is required")
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(PUBLIC_KEY)).verify(
            bytes.fromhex(packet["signature"]), signed_bytes(packet["body"]))
    except (InvalidSignature, ValueError, TypeError) as exc:
        raise ValueError("Parking-model signature does not match") from exc


def build_outcome(request, photo, packet):
    request, image_sha, input_hash = source_input(request, photo)
    _authenticate(packet)
    body = packet["body"]
    words = parking_instruction(request)
    if (not words or body.get("recipe") not in tuple(CALL_LIMITS) or body.get("models") != [QWEN_REVISION, SEG_REVISION]
            or body.get("input_sha256") != input_hash or body.get("image_sha256") != image_sha
            or body.get("words") != words):
        raise ValueError("Parking decision uses different models, frame or passenger request")
    classes = _decode_map(body["class_map"], request["scene"]["frame"])
    decision = replay_decision(request, classes, body["calls"], photo, body["recipe"])
    provenance = {"method": METHOD, "image": {"sha256": image_sha}, "request_sha256": digest(request),
                  "inference_input_sha256": input_hash, "parking_evidence": packet,
                  "reason": decision["reason"], "track_ids": decision.get("pair", []),
                  "independent_model_proposal": decision.get("result", {}).get("status") == "proposed"}
    out = {"request_id":request["request_id"], "provenance":provenance}
    if "error" in decision:
        return {**out, "ok":False, "error":{"code":"invalid_model_output", "message":decision["error"]}}
    return {**out, "ok":True, "result":decision["result"]}


def verify(request, outcome, *, artifacts=None):
    frame = request["scene"]["frame"]
    validated = build_outcome(request, Path(frame["image_path"]).read_bytes(),
                              outcome.get("provenance", {}).get("parking_evidence"))
    if validated != outcome:
        raise ValueError("Passenger parking outcome differs from its signed input and model decisions")
    return {"input_verified":True, "decision_reproduced":True, "model_images_reproduced":True,
            "independent_model_proposal":validated["provenance"]["independent_model_proposal"]}


def portable_request(request):
    clean = validate_request(request)
    scene = clean["scene"]
    scene["frame"]["image_path"] = "original.frame"
    scene["provenance"] = {"image_sha256":scene["provenance"]["image_sha256"]}
    return clean


class ModalParkingBackend:
    def __init__(self, python=None):
        root = Path(__file__).resolve().parents[2]
        if python is None:
            local = root / ".venv-modal" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            python = os.environ.get("TRACKRAG_MODAL_PYTHON") or (local if local.is_file() else sys.executable)
        self.python = str(python)
        self.root = root

    def interpret(self, request):
        path = Path(request["scene"]["frame"]["image_path"])
        if path.stat().st_size > MAX_IMAGE_BYTES:
            raise ValueError("Photograph exceeds the parking service input size")
        photo = path.read_bytes()
        request, _, _ = source_input(request, photo)
        payload = {"request":portable_request(request), "image":base64.b64encode(photo).decode("ascii"),
                   "recipe":INSPECTED_RECIPE}
        proc = subprocess.run([self.python, "-m", "tools.passenger_parking_call"],
                              input=signed_bytes(payload), capture_output=True, cwd=self.root, timeout=240,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if proc.returncode:
            raise RuntimeError("Parking GPU unavailable or the request could not be processed")
        packet = json.loads(proc.stdout)
        outcome = build_outcome(request, photo, packet)
        if packet["body"]["recipe"] != INSPECTED_RECIPE:
            raise ValueError("The deployed parking model is running a different live recipe")
        return outcome


def make_backend(name, *, artifacts=None):
    if name != "modal":
        raise ValueError("Unsupported passenger parking backend")
    return ModalParkingBackend()
