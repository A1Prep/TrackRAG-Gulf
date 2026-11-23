"""Private GPU transport and locally replayable evidence for visible parking places."""
import base64
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit

from PIL import Image

from . import place_grounding as place
from .parking_evidence import (MAX_IMAGE_BYTES, ModalParkingBackend, _authenticate, _decode_map,
                               signed_bytes, source_input)
from .passenger import digest
from .contracts import error_outcome
from tools.passenger_parking_study import SEG_REVISION
from tools.passenger_place_call import UNAVAILABLE

MODELS = {"base":place.BASE,"processor":place.PROCESSOR,"adapter_sha256":place.ADAPTER,
          "parking_segmenter":SEG_REVISION}
FINGERPRINTS = {"base_files":"13917d7d3d9b5d7a44775a6744e756b552bc4065bf73d633aa14278a0ac66417",
                "adapter_files":"6abadf2dd704566e2e1dcf24fb003ba1d7e044d88ee5d5aae9c279e9d573a3e7",
                "processor_files":"3393482a8a7c6da1ca1fa7eafc95b18b5c5539d72f0fe38b3c8830a5a2a69128"}

def unavailable(request, image_sha, input_sha):
    return error_outcome("place_service_unavailable", "The place model could not finish this request.",
                         request["request_id"], {"method":UNAVAILABLE, "request_sha256":digest(request),
                         "inference_input_sha256":input_sha, "image":{"sha256":image_sha}})


def verify_unavailable(request, outcome):
    """Verify the request and original that may receive a separate passenger mark."""
    path = Path(request["scene"]["frame"]["image_path"])
    if path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError("Photograph is too large")
    clean, sha, input_sha = source_input(request, path.read_bytes())
    if not place.place_reference(clean) or outcome != unavailable(clean, sha, input_sha):
        raise ValueError("Place service failure does not belong to this instruction and original")
    return {"input_verified":True, "independent_model_proposal":False}


def build_outcome(request, photo, packet):
    request, image_sha, input_sha = source_input(request, photo)
    context = place.current_context(request)
    _authenticate(packet)
    body = packet["body"]
    if (not context["words"] or body.get("recipe") != place.METHOD or body.get("models") != MODELS
            or body.get("input_sha256") != input_sha or body.get("context_sha256") != digest(context)
            or body.get("image_sha256") != image_sha):
        raise ValueError("Place model response does not match the current photograph, instruction and boxes")
    classes = _decode_map(body["class_map"], request["scene"]["frame"])
    calls = body["calls"]
    if not isinstance(calls, list) or len(calls) > 2:
        raise ValueError("Invalid number of place model responses")
    index = 0

    def replay(kind, guide, pixels):
        nonlocal index
        expected = place.call_identity(kind, guide, context["words"], pixels)
        if index >= len(calls) or any(calls[index].get(k) != v for k,v in expected.items()):
            raise ValueError("Place model input does not reproduce from the current photograph")
        answer = calls[index]["response"]
        index += 1
        return answer

    with Image.open(BytesIO(photo)) as source:
        result, reason = place.apply_steps(context, classes, source, replay)
    if index != len(calls):
        raise ValueError("Unused place model evidence")
    return {"ok":True, "request_id":request["request_id"], "result":result,
            "provenance":{"method":place.METHOD, "image":{"sha256":image_sha}, "request_sha256":digest(request),
                          "inference_input_sha256":input_sha,"place_evidence":packet,"reason":reason,
                          "independent_model_proposal":result["status"] == "proposed"}}


def verify(request, outcome):
    photo = Path(request["scene"]["frame"]["image_path"]).read_bytes()
    packet = outcome["provenance"]["place_evidence"]
    if build_outcome(request, photo, packet) != outcome:
        raise ValueError("Place result differs from the signed current model calls")
    return {"input_verified":True, "decision_reproduced":True, "model_images_reproduced":True,
            "language_calls":len(packet["body"]["calls"]), "surface_model_calls":1,
            "independent_model_proposal":outcome["provenance"]["independent_model_proposal"]}


class PublicPlaceBackend(ModalParkingBackend):
    def __init__(self, sources, python=None):
        super().__init__(python)
        self.allowed = set()
        for path in sources:
            for row in json.loads(Path(path).read_bytes())["photos"]:
                link = urlsplit(row["photo_url"])
                photo = re.fullmatch(r"/details/([1-9]\d*)/(\d+)/?", link.path)
                if (not photo or link.scheme != "https" or link.netloc != "kartaview.org" or link.query
                        or link.fragment or not re.fullmatch("[a-f0-9]{64}",row["sha256"])):
                    raise ValueError("Expected an original KartaView photo URL and SHA-256")
                self.allowed.add((row["sha256"],"kartaview_"+photo[1],int(photo[2])))

    def eligible(self, request):
        scene = request["scene"]
        f = scene["frame"]
        return (scene["source"] == "predicted" and
                (scene["provenance"].get("image_sha256"), f["recording_id"], f["frame_id"]) in self.allowed and
                bool(place.place_reference(request)))

    def interpret(self, request):
        if not self.eligible(request):
            raise ValueError("Only the selected public photographs can use hosted place grounding")
        path = Path(request["scene"]["frame"]["image_path"])
        if path.stat().st_size > MAX_IMAGE_BYTES:
            raise ValueError("Photograph is too large")
        photo = path.read_bytes()
        request, image_sha, input_sha = source_input(request, photo)
        payload = {"context":place.current_context(request), "input_sha256":input_sha, "photo":base64.b64encode(photo).decode("ascii")}
        try:
            proc = subprocess.run([self.python,"-m","tools.passenger_place_call"],input=signed_bytes(payload),
                                  cwd=self.root,capture_output=True,timeout=240,
                                  creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        except (subprocess.TimeoutExpired, OSError):
            return unavailable(request, image_sha, input_sha)
        if proc.returncode:
            return unavailable(request, image_sha, input_sha)
        return build_outcome(request, photo, json.loads(proc.stdout))


class PlaceDialogueGrounder:
    """Keep existing dialogue and passenger controls and replace eligible place interpretations only."""
    def __init__(self, model, backend):
        self.dialogue, self.backend = model, backend
        self.model = getattr(model, "model", model)
        from .single_vehicle_backend import SingleVehicleBackend
        self.single_vehicle = SingleVehicleBackend(backend) if isinstance(backend, PublicPlaceBackend) else None

    def __getattr__(self, name):
        return getattr(self.dialogue, name)

    def interpret(self, request):
        if self.single_vehicle is not None and self.single_vehicle.eligible(request):
            return self.single_vehicle.interpret(request)
        if self.backend.eligible(request):
            return self.backend.interpret(request)
        return self.dialogue.interpret(request)
