"""Private GPU transport and locally replayable evidence for visible parking places."""
import base64
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import subprocess
from tempfile import NamedTemporaryFile
from time import monotonic
from urllib.parse import urlsplit

from PIL import Image

from . import manual_surfaces, place_grounding as place
from .parking_evidence import (MAX_IMAGE_BYTES, ModalParkingBackend, _authenticate, _decode_map,
                               signed_bytes, source_input)
from .passenger import digest, unresolved
from .place_lookup import PlaceLookups
from .contracts import error_outcome
from tools.passenger_parking_study import SEG_REVISION
from tools.passenger_place_call import UNAVAILABLE

MODELS = {"base":place.BASE,"processor":place.PROCESSOR,"adapter_sha256":place.ADAPTER,
          "parking_segmenter":SEG_REVISION}
FINGERPRINTS = {"base_files":"13917d7d3d9b5d7a44775a6744e756b552bc4065bf73d633aa14278a0ac66417",
                "adapter_files":"6abadf2dd704566e2e1dcf24fb003ba1d7e044d88ee5d5aae9c279e9d573a3e7",
                "processor_files":"3393482a8a7c6da1ca1fa7eafc95b18b5c5539d72f0fe38b3c8830a5a2a69128"}
WAIT_METHOD = "public_place_lookup_wait_v1"
_native_guard = ContextVar("public_photo_native_fallback",default=nullcontext)
INTERACTIVE_WAIT_SECONDS = 4.0
WAIT_MESSAGES = {
    "pending":"I’m still checking this photo. You can mark your spot now or send the same description again in a moment.",
    "busy":"The photo checker is busy. You can mark your spot now or try this description again shortly.",
}


class PlaceTransportUnavailable(RuntimeError):
    pass


@contextmanager
def guard_public_fallback(guard):
    """A public worker can read cached evidence freely, but an actual laptop fallback shares the model lock."""
    token = _native_guard.set(guard)
    try:
        yield
    finally:
        _native_guard.reset(token)


def waiting_outcome(request, image_sha, input_sha, status):
    return {"ok":True, "request_id":request["request_id"], "result":unresolved(WAIT_MESSAGES[status]),
            "provenance":{"method":WAIT_METHOD, "reason":status, "operational_status":status,
                          "request_sha256":digest(request), "inference_input_sha256":input_sha,
                          "image":{"sha256":image_sha},"independent_model_proposal":False}}

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
    path = Path(request["scene"]["frame"]["image_path"])
    if path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError("Photograph is too large")
    photo = path.read_bytes()
    if outcome["provenance"].get("method") == WAIT_METHOD:
        clean, image_sha, input_sha = source_input(request,photo)
        status = outcome["provenance"].get("operational_status")
        if (not place.place_reference(clean) or not isinstance(status,str) or status not in WAIT_MESSAGES
                or outcome != waiting_outcome(clean,image_sha,input_sha,status)):
            raise ValueError("Place lookup status does not belong to this instruction and original")
        return {"input_verified":True, "decision_reproduced":False, "model_images_reproduced":False,
                "language_calls":0, "current_language_calls":0, "surface_model_calls":0,
                "lookup_status":status, "independent_model_proposal":False}
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
        self.lookups = PlaceLookups()
        self.allowed = set()
        for path in sources:
            for row in json.loads(Path(path).read_bytes())["photos"]:
                link = urlsplit(row["photo_url"])
                photo = re.fullmatch(r"/details/([1-9]\d*)/(\d+)/?", link.path)
                if (not photo or link.scheme != "https" or link.netloc != "kartaview.org" or link.query
                        or link.fragment or not re.fullmatch("[a-f0-9]{64}",row["sha256"])):
                    raise ValueError("Expected an original KartaView photo URL and SHA-256")
                self.allowed.add((row["sha256"],"kartaview_"+photo[1],int(photo[2])))

    def allows_photo(self, request):
        from .local_surfaces import local_only
        scene = request["scene"]
        f = scene["frame"]
        return (not local_only(scene) and scene["source"] == "predicted" and
                (scene["provenance"].get("image_sha256"), f["recording_id"], f["frame_id"]) in self.allowed)

    def eligible(self, request):
        return self.allows_photo(request) and bool(place.place_reference(request))

    @staticmethod
    def repeat_key(request, image_sha):
        from .data import inference_input_sha256
        return inference_input_sha256({**request,"request_id":"public-place-repeat"},image_sha)

    def cache_path(self, request, image_sha):
        return self.cache/(self.repeat_key(request,image_sha)+".json")

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
        deadline = monotonic() + INTERACTIVE_WAIT_SECONDS
        if not self.eligible(request):
            raise ValueError("Only the selected public photographs can use hosted place grounding")
        path = Path(request["scene"]["frame"]["image_path"])
        if path.stat().st_size > MAX_IMAGE_BYTES:
            raise ValueError("Photograph is too large")
        photo = path.read_bytes()
        request, image_sha, input_sha = source_input(request, photo)
        from . import whole_bay
        bay = whole_bay.automatic(request, photo)
        if bay is not None:
            return bay
        key = self.repeat_key(request,image_sha)
        if self.cache is not None:
            try:
                saved=self.cache/(key+".json")
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
        try:
            status, completed = self.lookups.lookup(key,lambda:self._fetch(request,photo,input_sha),deadline)
        except PlaceTransportUnavailable:
            return unavailable(request,image_sha,input_sha)
        if status != "complete":
            return waiting_outcome(request,image_sha,input_sha,status)
        # A prior turn can have a different request ID or local path. Replay its signed inputs and
        # keep exception/mutation handling local to this passenger's copy of the receipt.
        original = json.loads(completed)
        if original["provenance"]["request_sha256"] == digest(request):
            return original
        return build_outcome(request,photo,original["provenance"]["place_evidence"],
                             recorded_request_id=original["request_id"])

    def _fetch(self, request, photo, input_sha):
        payload = {"context":place.current_context(request), "input_sha256":input_sha, "photo":base64.b64encode(photo).decode("ascii")}
        try:
            proc = subprocess.run([self.python,"-m","tools.passenger_place_call"],input=signed_bytes(payload),
                                  cwd=self.root,capture_output=True,timeout=240,
                                  creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        except (subprocess.TimeoutExpired, OSError) as exc:
            raise PlaceTransportUnavailable("The public place service could not finish") from exc
        if proc.returncode:
            raise PlaceTransportUnavailable("The public place service could not finish")
        if len(proc.stdout) > 3_000_000:
            raise ValueError("Oversize public place receipt")
        return signed_bytes(self.remember(request,photo,json.loads(proc.stdout)))


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
        from . import bay_intent,curb_point_language,local_surfaces,native_destination,roadside_context,visual_space
        from . import vehicle_ground_reference
        if local_surfaces.local_only(request.get("scene",{})):
            return False
        return ((isinstance(self.backend,PublicPlaceBackend) and self.backend.allows_photo(request) and
                 (vehicle_ground_reference.direct_reference(request) is not None or bay_intent.parse(request["text"]) is not None or
                  curb_point_language.indexed_reference(request["text"],expanded=True) or
                  (curb_point_language.abbreviated_pair(request["text"]) is not None and
                   roadside_context.get_packet(request["scene"]["provenance"]["image_sha256"]) is not None and
                   manual_surfaces.get_packet(request["scene"]["provenance"]["image_sha256"]) is not None) or
                  (native_destination._parts(request["text"]) is not None and
                   visual_space.get_extension_packet(request["scene"]["provenance"]["image_sha256"]) is not None) or
                  (re.search(r"\b(?:behind|(?:left|right) of)\b",request["text"],re.I) and
                   visual_space.get_packet(request["scene"]["provenance"]["image_sha256"],paved=True) is not None))) or
                (self.single_vehicle is not None and self.single_vehicle.eligible(request)) or self.backend.eligible(request))

    def inspect_requested_bay(self, request):
        from . import observed_pullin,photo_clarification,whole_bay
        from .local_surfaces import matches_scene
        local=matches_scene(request.get("scene",{}))
        if local or (isinstance(self.backend,PublicPlaceBackend) and self.backend.allows_photo(request)):
            prior=observed_pullin.supplement(request,whole_bay.explicit_request(request))
            if local or (prior is not None and (photo_clarification.vegetation_reference(request["text"]) or
                          photo_clarification.has_public_alternative(request,prior) or
                          prior["provenance"].get("reason") in
                          {"indexed_reference_unmatched","indexed_no_open_location","parking_fragments_at_vehicles"})):
                try:
                    return photo_clarification.explain(request,prior,local=local)
                except (ValueError,TypeError,KeyError,OSError):
                    pass
            return prior
        return None

    def interpret(self, request):
        from . import bay_intent, single_vehicle, vehicle_ground_reference, vehicle_wording
        from .local_surfaces import local_only
        agreed = vehicle_wording.reply(request)
        if agreed and agreed[0] == "accepted":
            # This is consent to check an explicitly named vehicle. Its ground still needs source evidence.
            result = vehicle_ground_reference.available(request) if not local_only(request["scene"]) else None
            return result if result is not None else vehicle_ground_reference.unavailable(request)
        if local_only(request.get("scene",{})):
            bay = self.inspect_requested_bay(request)
            if bay is not None:
                return bay
            with _native_guard.get()():
                return self.dialogue.interpret(request)
        if isinstance(self.backend, PublicPlaceBackend) and self.backend.allows_photo(request):
            ref = vehicle_ground_reference.direct_reference(request)
            if ref and (ref.get('qualified') or ref['relation'] == 'beside'):
                pictured = vehicle_ground_reference.available(request)
                return pictured if pictured is not None else vehicle_ground_reference.unavailable(request)
        # A direct request behind one identified vehicle already has its own signed
        # rear-view and parking check. Do not scan unrelated storefronts first.
        if self.single_vehicle is not None and self.single_vehicle.eligible(request):
            ref = single_vehicle.request_reference(request)
            prefix = re.split(r"\bbehind\b",request["text"].casefold(),maxsplit=1)[0]
            simple = (ref and not ref["unsupported"] and ref["relation"] == "behind" and
                      set(re.findall(r"[a-z]+",prefix)) <= bay_intent.FILLER | {"off","offload","get","there"} and
                      bay_intent.permitted_destination(request["text"]) and not bay_intent.PARKING.search(request["text"]))
            if simple:
                return self.single_vehicle.interpret(request)
        bay=self.inspect_requested_bay(request)
        if bay is not None:
            return bay
        if self.single_vehicle is not None and self.single_vehicle.eligible(request):
            return self.single_vehicle.interpret(request)
        if self.backend.eligible(request):
            outcome=self.backend.interpret(request)
            from . import photo_clarification
            if (isinstance(self.backend,PublicPlaceBackend) and isinstance(outcome,dict) and
                    outcome.get("result",{}).get("status")=="needs_clarification" and
                    photo_clarification.vegetation_reference(request["text"])):
                try:
                    return photo_clarification.explain(request,outcome)
                except (OSError,ValueError,TypeError,KeyError):
                    return outcome
            return outcome
        with _native_guard.get()():
            return self.dialogue.interpret(request)
