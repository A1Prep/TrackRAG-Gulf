"""Private frontage evidence service using versioned specialists and the trained place reader."""
import base64
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from time import time

import modal

from tools.passenger_frontage_call import APP, RECIPE, require_releases
from tools.passenger_place_modal import TrainedPlaceReader, image


app = modal.App(APP)
POINT_APP = "trackrag-public-frontage-pointing-trial"
SURFACE_APP = "trackrag-frontage-language-masks"
POINT_VERSION = SURFACE_VERSION = 1
POINT_REVISION = "e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"
SURFACE_REVISION = "999e0328d9e10b484360c477313983f9afdd7050"
JOBS = modal.Dict.from_name("trackrag-frontage-signed-jobs-v1",create_if_missing=True)
if modal.is_local():
    folder = Path(__file__).resolve().parent
    for name in ("passenger_frontage_call", "passenger_frontage_pointing", "passenger_frontage_surfaces", "passenger_place_modal"):
        image = image.add_local_file(folder/(name+".py"), "/root/tools/"+name+".py")


def validate_input(context, input_sha, photo):
    from PIL import Image
    from trackrag.dropoff.contracts import validate_box
    from trackrag.dropoff.parking_evidence import MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS
    from trackrag.dropoff.passenger import digest

    if (not isinstance(context,dict) or set(context) != {"words","size","current_boxes","image_sha256"}
            or not isinstance(photo,bytes) or not 0 < len(photo) <= MAX_IMAGE_BYTES
            or sha256(photo).hexdigest() != context["image_sha256"]
            or not isinstance(context["words"],str) or not 1 <= len(context["words"]) <= 1600
            or not isinstance(context["current_boxes"],list) or len(context["current_boxes"])>3000
            or not isinstance(context["size"],list) or len(context["size"]) != 2
            or any(type(n) is not int or n<=0 for n in context["size"])
            or input_sha != digest({"recipe":RECIPE,"context":context})):
        raise ValueError("Frontage request does not match its photo, current context and recipe")
    with Image.open(BytesIO(photo)) as original:
        if list(original.size)!=context["size"] or original.width*original.height>MAX_IMAGE_PIXELS:
            raise ValueError("Unexpected original photograph dimensions")
        original.verify()
    for box in context["current_boxes"]:
        validate_box(box,*context["size"])


def point_evidence(raw, key, words):
    from tools.passenger_frontage_pointing import coordinates

    fields = {"image_sha256","words","revision","point_answer","point_1000","check"}
    if (not isinstance(raw,dict) or set(raw)!=fields or raw["image_sha256"]!=key
            or raw["words"]!=words or raw["revision"]!=POINT_REVISION
            or not isinstance(raw["point_answer"],str) or len(raw["point_answer"])>4096):
        raise ValueError("Unexpected versioned pointing response")
    point = coordinates(raw["point_answer"])
    if (raw["point_1000"]!=point or (point is None and raw["check"] is not None)
            or (point is not None and (not isinstance(raw["check"],str) or len(raw["check"])>4096))):
        raise ValueError("Model's point differs from its original answer")
    return {**raw,"point_1000":point}


def validate_maps(packed, key, words):
    import numpy as np
    from tools.passenger_frontage_surfaces import QUERIES, landmark

    if not isinstance(packed,bytes) or not 0 < len(packed) <= 512 * 1024:
        raise ValueError("Invalid surface evidence size")
    with np.load(BytesIO(packed),allow_pickle=False) as data:
        if (set(data.files)!={"logits","revision","source","words","queries"} or str(data["source"])!=key
                or str(data["words"])!=words or str(data["revision"])!=SURFACE_REVISION
                or data["queries"].tolist()!=[*QUERIES,landmark(words)]):
            raise ValueError("Surface maps belong to a different photograph, checkpoint or instruction")
        logits = data["logits"]
        if logits.shape!=(8,96,144) or logits.dtype!=np.float16 or not np.isfinite(logits).all():
            raise ValueError("Unexpected surface model logits")


@app.cls(image=image,gpu="A100-40GB",cpu=4,memory=30000,timeout=570,startup_timeout=360,
         min_containers=0,max_containers=1,scaledown_window=30,retries=0,
         volumes={"/base":modal.Volume.from_name("trackrag-qwen8b-public-weights").read_only(),
                  "/parking":modal.Volume.from_name("trackrag-parking-vistas-weights").read_only()},
         secrets=[modal.Secret.from_name("trackrag-parking-receipt-v1")])
class FrontageEvidence(TrainedPlaceReader):
    @modal.enter()
    def load(self):
        super().load()
        self.pointer = modal.Cls.from_name(POINT_APP,"PointingTrial",environment_name="main")()
        cls = modal.Cls.from_name(SURFACE_APP,"Surfaces",environment_name="main")
        self.surfaces = cls.with_options(gpu="A10G")()
        self.alternate_surfaces = cls()

    def remember_jobs(self,input_sha,point_call,maps_call,accelerator):
        from trackrag.dropoff.parking_evidence import signed_bytes
        body = {"recipe":"frontage_private_job_ticket_v1","input_sha256":input_sha,"created":int(time()),
                "point_job":point_call.object_id,"surface_job":maps_call.object_id,"accelerator":accelerator}
        JOBS.put(input_sha,{"body":body,"signature":self.key.sign(signed_bytes(body)).hex()})

    def jobs_for(self,input_sha,photo,key,words):
        from trackrag.dropoff.parking_evidence import _authenticate

        # Store only authenticated, short-lived worker IDs. A retry rejoins the
        # original public-photo jobs even when one accelerator queued past its wait.
        ticket = JOBS.get(input_sha)
        if ticket is not None:
            _authenticate(ticket)
            body = ticket["body"]
            if body.get("recipe")!="frontage_private_job_ticket_v1" or body.get("input_sha256")!=input_sha:
                raise ValueError("Unexpected private frontage job ticket")
            if 0 <= time()-body["created"] < 24*3600:
                return (modal.FunctionCall.from_id(body["point_job"]),modal.FunctionCall.from_id(body["surface_job"]),
                        body["accelerator"])
        point_call = self.pointer.predict.spawn(photo,key,words)
        maps_call = self.surfaces.compute.spawn(photo,key,words)
        self.remember_jobs(input_sha,point_call,maps_call,"A10G")
        return point_call,maps_call,"A10G"

    @modal.method()
    def collect(self, context, input_sha, photo):
        from trackrag.dropoff.parking_evidence import signed_bytes
        from trackrag.dropoff.passenger import digest

        validate_input(context,input_sha,photo)
        releases = ((POINT_APP,POINT_VERSION),(SURFACE_APP,SURFACE_VERSION))
        require_releases(*releases)
        key, words = context["image_sha256"], context["words"]
        # Independent workers see only the original photo, its hash and the words.
        # Only this trusted reader can sign the assembled response.
        point_call, maps_call, accelerator = self.jobs_for(input_sha,photo,key,words)
        try:
            point = point_evidence(point_call.get(timeout=405),key,words)
            try:
                packed = maps_call.get(timeout=80)
            except TimeoutError:
                if accelerator != "A10G":
                    raise
                maps_call.cancel()
                maps_call = self.alternate_surfaces.compute.spawn(photo,key,words)
                self.remember_jobs(input_sha,point_call,maps_call,"L4")
                packed = maps_call.get(timeout=90)
            validate_maps(packed,key,words)
        except TimeoutError:
            raise
        except Exception:
            JOBS.pop(input_sha,None)
            raise
        checks = None
        if point["point_1000"] is not None:
            checks = self.check_frontage_point(photo,key,words,point["point_1000"])
        require_releases(*releases)
        body = {"recipe":RECIPE, "input_sha256":input_sha, "context_sha256":digest(context), "image_sha256":key,
                "point":point, "checks":checks, "map_sha256":sha256(packed).hexdigest(),
                "map_npz":base64.b64encode(packed).decode("ascii")}
        return {"body":body,"signature":self.key.sign(signed_bytes(body)).hex()}
