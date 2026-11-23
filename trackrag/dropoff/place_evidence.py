"""Private GPU transport and locally replayable evidence for visible parking places."""
import base64
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import subprocess
from tempfile import NamedTemporaryFile
from urllib.parse import urlsplit

from PIL import Image

from . import manual_surfaces, place_grounding as place
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


def build_outcome(request, photo, packet, *, recorded_request_id=None):
    request, image_sha, input_sha = source_input(request, photo)
    context = place.current_context(request)
    _authenticate(packet)
    body = packet["body"]
    recorded_input = input_sha
    if recorded_request_id is not None:
        # A repeated browser turn gets a new ID. Every other inference input must still match the signed run.
        from .data import inference_input_sha256
        earlier = {**request,"request_id":recorded_request_id}
        recorded_input = inference_input_sha256(earlier,image_sha)
    if (not context["words"] or body.get("recipe") != place.METHOD or body.get("models") != MODELS
            or body.get("input_sha256") != recorded_input or body.get("context_sha256") != digest(context)
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
    replay = ({"reused_source_request_id":recorded_request_id,"recorded_inference_input_sha256":recorded_input}
              if recorded_request_id is not None and recorded_request_id!=request["request_id"] else {})
    return {"ok":True, "request_id":request["request_id"], "result":result,
            "provenance":{"method":place.METHOD, "image":{"sha256":image_sha}, "request_sha256":digest(request),
                          "inference_input_sha256":input_sha,"place_evidence":packet,"reason":reason,
                          "independent_model_proposal":result["status"] == "proposed", **replay}}


SURFACE_ONLY = place.EMPTY_SURFACE_METHOD


def empty_surface_outcome(request, photo, packet):
    """Reuse the same visual first stage for a new wording when it cannot supply a candidate."""
    clean, image_sha, input_sha = source_input(request, photo)
    context = place.current_context(clean)
    if not context["words"]:
        raise ValueError("This instruction does not request a named place.")
    classes = manual_surfaces.decode(packet, clean["scene"]["frame"], image_sha)
    if place.surface_options(classes, context["size"], context["current_boxes"]):
        return None
    result, reason = place.no_parking_surface()
    return {"ok":True, "request_id":clean["request_id"], "result":result, "provenance":{
        "method":SURFACE_ONLY, "request_sha256":digest(clean), "image":{"sha256":image_sha},
        "inference_input_sha256":input_sha, "reason":reason, "surface_evidence":packet,
        "independent_model_proposal":False}}


def verify(request, outcome):
    photo = Path(request["scene"]["frame"]["image_path"]).read_bytes()
    if outcome["provenance"].get("method") == SURFACE_ONLY:
        if empty_surface_outcome(request, photo, outcome["provenance"]["surface_evidence"]) != outcome:
            raise ValueError("The signed parking map does not explain this clarification.")
        return {"input_verified":True, "decision_reproduced":True, "model_images_reproduced":False,
                "language_calls":0, "current_language_calls":0, "surface_model_calls":0,
                "signed_surface_replayed":True, "independent_model_proposal":False}
    packet = outcome["provenance"]["place_evidence"]
    original_id=outcome["provenance"].get("reused_source_request_id")
    if build_outcome(request, photo, packet, recorded_request_id=original_id) != outcome:
        raise ValueError("Place result differs from the signed current model calls")
    return {"input_verified":True, "decision_reproduced":True, "model_images_reproduced":True,
            "language_calls":len(packet["body"]["calls"]), "surface_model_calls":1,
            "independent_model_proposal":outcome["provenance"]["independent_model_proposal"],
            "reused_model_evidence":original_id is not None,"current_language_calls":0 if original_id else len(packet["body"]["calls"])}


class PublicPlaceBackend(ModalParkingBackend):
    def __init__(self, sources, python=None, *, cache=None):
        super().__init__(python)
        self.cache=Path(cache) if cache is not None else None
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

    def cache_path(self, request, image_sha):
        from .data import inference_input_sha256
        context = inference_input_sha256({**request,"request_id":"public-place-repeat"},image_sha)
        return self.cache/(context+".json")

    def remember(self, request, photo, packet):
        """Cache only authenticated responses for an authorized public input."""
        if not self.eligible(request):
            raise ValueError("This photograph or instruction is outside the public place scope")
        verified=build_outcome(request,photo,packet)
        if self.cache is None:
            return verified
        path=self.cache_path(request,verified["provenance"]["image"]["sha256"])
        temporary=None
        try:
            path.parent.mkdir(parents=True,exist_ok=True)
            with NamedTemporaryFile(dir=path.parent,delete=False) as tmp:
                temporary=Path(tmp.name)
                tmp.write(signed_bytes({"request_id":request["request_id"],"packet":packet}))
            os.replace(temporary,path)
        except OSError:
            pass  # A verified first response remains usable even on a read-only cache.
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
        return verified

    def interpret(self, request):
        if not self.eligible(request):
            raise ValueError("Only the selected public photographs can use hosted place grounding")
        path = Path(request["scene"]["frame"]["image_path"])
        if path.stat().st_size > MAX_IMAGE_BYTES:
            raise ValueError("Photograph is too large")
        photo = path.read_bytes()
        request, image_sha, input_sha = source_input(request, photo)
        if self.cache is not None:
            try:
                saved=self.cache_path(request,image_sha)
                if saved.stat().st_size<=3_000_000:
                    receipt=json.loads(saved.read_bytes())
                    return build_outcome(request,photo,receipt["packet"],recorded_request_id=receipt["request_id"])
            except (OSError,ValueError,TypeError,KeyError):
                pass
        try:
            surface = manual_surfaces.get_packet(image_sha)
            if surface is not None:
                empty = empty_surface_outcome(request, photo, surface)
                if empty is not None:
                    return empty
        except (OSError, ValueError, TypeError, KeyError):
            pass  # The normal trained route can obtain fresh evidence for a missing or damaged map.
        payload = {"context":place.current_context(request), "input_sha256":input_sha, "photo":base64.b64encode(photo).decode("ascii")}
        try:
            proc = subprocess.run([self.python,"-m","tools.passenger_place_call"],input=signed_bytes(payload),
                                  cwd=self.root,capture_output=True,timeout=240,
                                  creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        except (subprocess.TimeoutExpired, OSError):
            return unavailable(request, image_sha, input_sha)
        if proc.returncode:
            return unavailable(request, image_sha, input_sha)
        return self.remember(request,photo,json.loads(proc.stdout))


class PlaceDialogueGrounder:
    """Keep existing dialogue and passenger controls and replace eligible place interpretations only."""
    def __init__(self, model, backend):
        self.dialogue, self.backend = model, backend
        self.model = getattr(model, "model", model)
        from .single_vehicle_backend import SingleVehicleBackend
        self.single_vehicle = SingleVehicleBackend(backend) if isinstance(backend, PublicPlaceBackend) else None

    def __getattr__(self, name):
        return getattr(self.dialogue, name)

    def uses_public_service(self, request):
        return (self.single_vehicle is not None and self.single_vehicle.eligible(request)) or self.backend.eligible(request)

    def interpret(self, request):
        if self.single_vehicle is not None and self.single_vehicle.eligible(request):
            return self.single_vehicle.interpret(request)
        if self.backend.eligible(request):
            return self.backend.interpret(request)
        return self.dialogue.interpret(request)
