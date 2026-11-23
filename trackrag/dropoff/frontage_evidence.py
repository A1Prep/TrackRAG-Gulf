"""Replay confidence-gated public frontage proposals against an original photo and request."""
import base64
from concurrent.futures import Future, TimeoutError as FutureTimeout
from hashlib import sha256
from io import BytesIO
import json
import logging
import os
from pathlib import Path
import subprocess
import tempfile
from threading import Lock, Thread
from time import monotonic
from zipfile import BadZipFile, ZipFile

import numpy as np
from PIL import Image

from tools import passenger_frontage_confidence as confidence
from . import frontage_verifier, place_evidence, place_grounding
from .parking_evidence import MAX_IMAGE_BYTES, _authenticate, signed_bytes, source_input
from .passenger import digest, proposed

METHOD = "molmo2_frontage_confidence_v1"
MODEL_HASH = "e01b87af952230391df6496f963faf214a1e187121f196d1d0b314f51a9d3fcb"
INTERACTIVE_WAIT_SECONDS = 5
MODEL = Path(__file__).resolve().parents[2]/"configs/dropoff/passenger-frontage-confidence-v1.json"
LOG = logging.getLogger(__name__)


def context_for(request):
    frame=request["scene"]["frame"]
    return {"words":request["text"], "size":[frame["width"],frame["height"]],
            "image_sha256":request["scene"]["provenance"]["image_sha256"],
            "current_boxes":[t["bbox_xyxy"] for t in request["scene"]["tracks"] if t["frame_id"]==frame["frame_id"]]}


def prior_eligible(request, prior):
    if (not isinstance(prior,dict) or prior.get("request_id")!=request["request_id"] or request["dialogue"]
            or request["scene"]["source"]!="predicted" or not frontage_verifier.describes_place(request)):
        return False
    source=prior.get("provenance",{})
    # Only a signed place response can prove the prior decision during journal replay.
    if source.get("method") not in {place_grounding.METHOD, place_evidence.SURFACE_ONLY}:
        return False
    if (source.get("image",{}).get("sha256")!=request["scene"]["provenance"]["image_sha256"]
            or source.get("request_sha256")!=digest(request)):
        return False
    return prior.get("ok") is True and prior.get("result",{}).get("status") in {
        "insufficient_visual_evidence","needs_clarification"}


def model_input(context):
    return digest({"recipe":METHOD,"context":context})


def _scored(request, photo, packet):
    ctx=context_for(request)
    _authenticate(packet)
    body=packet["body"]
    if (body.get("recipe")!=METHOD or body.get("image_sha256")!=sha256(photo).hexdigest()
            or body.get("context_sha256")!=digest(ctx) or body.get("input_sha256")!=model_input(ctx)):
        raise ValueError("Frontage evidence does not belong to the current image, request and observations")
    point=body["point"]
    if not isinstance(point,dict) or point.get("image_sha256")!=ctx["image_sha256"] or point.get("words")!=ctx["words"]:
        raise ValueError("Point source or instruction changed")
    raw=body["map_npz"]
    if not isinstance(raw,str) or len(raw)>1_300_000:
        raise ValueError("Invalid frontage surface packet")
    packed=base64.b64decode(raw,validate=True)
    if sha256(packed).hexdigest()!=body["map_sha256"]:
        raise ValueError("Frontage surface packet changed")
    try:
        with ZipFile(BytesIO(packed)) as z:
            entries=z.infolist()
            if (len(entries)!=5 or {i.filename for i in entries}!={"logits.npy","revision.npy","source.npy","words.npy","queries.npy"}
                    or sum(i.file_size for i in entries)>900_000):
                raise ValueError("Unexpected frontage array bundle")
    except BadZipFile as exc:
        raise ValueError("Invalid frontage array bundle") from exc
    check=body["checks"]
    if point["point_1000"] is not None:
        _authenticate(check)
        calls=check["body"]["calls"]
        if len(calls)!=2:
            raise ValueError("Missing named-place or ground-surface check")
        with Image.open(BytesIO(photo)) as source:
            original=source.convert("RGB")
        for call,(kind,guide,coord) in zip(calls,(("scene",frontage_verifier.SCENE,None),
                ("surface",frontage_verifier.SURFACE,point["point_1000"])),strict=True):
            picture=frontage_verifier.photo_view(original,coord)
            ident=place_grounding.call_identity(kind,guide,ctx["words"],picture)
            if any(call.get(k)!=v for k,v in ident.items()):
                raise ValueError("Trained verifier image or prompt cannot be reproduced")
    elif check is not None:
        raise ValueError("A missing specialist point cannot have prompted a crop check")
    weights=MODEL.read_bytes()
    if sha256(weights).hexdigest()!=MODEL_HASH:
        raise ValueError("Frontage confidence checkpoint changed")
    with np.load(BytesIO(packed),allow_pickle=False) as maps:
        features=confidence.evidence(request,point,maps,check)
    return confidence.predict(features,json.loads(weights)), point["point_1000"]


def build_outcome(request, photo, packet, prior):
    clean,source,request_input=source_input(request,photo)
    if not prior_eligible(clean,prior):
        raise ValueError("A frontage supplement requires the original eligible no-pin response")
    source_info = prior["provenance"]
    if source_info["method"] == place_evidence.SURFACE_ONLY:
        original = place_evidence.empty_surface_outcome(clean, photo, source_info["surface_evidence"])
    else:
        original=place_evidence.build_outcome(clean,photo,source_info["place_evidence"],
                                              recorded_request_id=source_info.get("reused_source_request_id"))
    if original!=prior:
        raise ValueError("Original place decision cannot be reproduced")
    decision,pt=_scored(clean,photo,packet)
    if not decision["accept"]:
        return prior
    frame=clean["scene"]["frame"];w,h=frame["width"],frame["height"]
    x,y=[min(dim-1,n*dim/1000) for dim,n in zip((w,h),pt)]
    # A small display outline surrounds a single estimated point. It measures no physical clearance.
    dx,dy=w*.008,h*.008
    result=proposed([x,y],[max(0.,x-dx),max(0.,y-dy),min(float(w),x+dx),min(float(h),y+dy)])
    return {"ok":True,"request_id":clean["request_id"],"result":result,
            "provenance":{"method":METHOD,"request_sha256":digest(clean),"inference_input_sha256":request_input,
                          "image":{"sha256":source},"independent_model_proposal":True,"confidence_sha256":MODEL_HASH,
                          "confidence_score":decision["score"],"reason":"verified_frontage_point",
                          "frontage_evidence":packet,"prior_outcome":prior}}


def verify(request, outcome):
    src=outcome["provenance"]
    p=Path(request["scene"]["frame"]["image_path"])
    if p.stat().st_size>MAX_IMAGE_BYTES or src.get("method")!=METHOD:
        raise ValueError("Invalid frontage proposal")
    if build_outcome(request,p.read_bytes(),src["frontage_evidence"],src["prior_outcome"])!=outcome:
        raise ValueError("The displayed frontage point or its decision evidence changed")
    # Two specialist calls and two trained scene/pavement classifications, in addition to any prior place calls.
    prior_evidence=src["prior_outcome"]["provenance"].get("place_evidence",{})
    count=4+len(prior_evidence.get("body",{}).get("calls",[]))
    return {"input_verified":True,"decision_reproduced":True,"model_images_reproduced":True,
            "language_calls":count,"independent_model_proposal":True,"confidence_sha256":MODEL_HASH}


class PublicFrontageGrounder:
    """Supplement eligible first-turn public place abstentions while retaining the original decision."""
    def __init__(self, dialogue, public_backend, sources, cache):
        self.dialogue=dialogue
        self.model=getattr(dialogue,"model",dialogue)
        self.public=public_backend
        self.sources=[Path(p).resolve() for p in sources]
        self.cache=Path(cache)
        self._pending={}
        self._pending_lock=Lock()

    def __getattr__(self,name):
        return getattr(self.dialogue,name)

    def interpret(self, request):
        from .local_surfaces import local_only
        deadline = monotonic() + INTERACTIVE_WAIT_SECONDS
        prior=self.dialogue.interpret(request)
        if local_only(request["scene"]):
            return prior
        frame=request["scene"]["frame"]
        key=(request["scene"]["provenance"].get("image_sha256"),frame["recording_id"],frame["frame_id"])
        if key not in self.public.allowed or not prior_eligible(request,prior):
            return prior
        try:
            return self._interpret(request,prior,deadline)
        except (ValueError,KeyError,TypeError,OSError,RuntimeError,subprocess.SubprocessError) as exc:
            # The already validated first response remains available during a service or cache failure.
            LOG.warning("Public frontage supplement unavailable (%s): %s",type(exc).__name__,str(exc)[:170])
            return prior

    def _interpret(self, request, prior, deadline):
        source=Path(request["scene"]["frame"]["image_path"])
        if source.stat().st_size>MAX_IMAGE_BYTES:
            raise ValueError("Public photo exceeds service limit")
        photo=source.read_bytes();clean,_,_=source_input(request,photo)
        ctx=context_for(clean);key=model_input(ctx)
        destination=self.cache/(key+".json")
        if destination.is_file() and destination.stat().st_size<3_000_000:
            try:
                return build_outcome(clean,photo,json.loads(destination.read_bytes()),prior)
            except (ValueError,KeyError,TypeError,OSError):
                LOG.warning("Discarded invalid frontage cache entry %s",key[:12])
        # Keep an already actionable clarification available while cold specialists load.
        # Equal requests share a job. Limit the number of distinct photos held in memory.
        with self._pending_lock:
            pending=self._pending.get(key)
            if pending is not None:
                # The first caller already waited for this optional second opinion. A repeated turn
                # can show its verified primary response while that same job finishes.
                return prior
            if len(self._pending)>=2:
                return prior
            pending=self._pending[key]=Future()
            Thread(target=self._collect, args=(clean,photo,key,pending), daemon=True).start()
        try:
            packet=pending.result(timeout=max(0.,deadline-monotonic()))
        except FutureTimeout:
            return prior
        return build_outcome(clean,photo,packet,prior)

    def _collect(self, request, photo, key, pending):
        try:
            packet=self._fetch(request,photo,key)
        except Exception as exc:
            LOG.warning("Could not obtain verified frontage evidence (%s)",type(exc).__name__)
            pending.set_exception(exc)
        else:
            pending.set_result(packet)
        finally:
            with self._pending_lock:
                self._pending.pop(key,None)

    def _fetch(self, request, photo, key):
        ctx=context_for(request)
        destination=self.cache/(key+".json")
        command=[self.public.python,"-m","tools.passenger_frontage_call"]
        for manifest in self.sources:command.extend(["--sources",str(manifest)])
        payload={"context":ctx,"input_sha256":key,"photo":base64.b64encode(photo).decode("ascii")}
        proc=subprocess.run(command,input=signed_bytes(payload),capture_output=True,cwd=self.public.root,
                            timeout=620,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        if proc.returncode or len(proc.stdout)>3_000_000:
            raise RuntimeError("The private frontage service did not return a complete result")
        packet=json.loads(proc.stdout)
        _scored(request,photo,packet)
        temporary=None
        try:
            self.cache.mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=self.cache,delete=False) as tmp:
                temporary=Path(tmp.name)
                tmp.write(proc.stdout)
            os.replace(temporary,destination)
        except OSError:
            LOG.warning("Could not cache a verified public frontage result")
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    LOG.warning("Could not remove an unused frontage cache file")
        return packet
