"""Private vehicle rear-view checker using the same trained adapter and semantic segmenter as parking."""
import base64
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re
import zlib

import modal
from tools.passenger_place_modal import TrainedPlaceReader,image
from tools.passenger_single_vehicle_call import APP

app=modal.App(APP)
if modal.is_local():
    root=Path(__file__).resolve().parents[1]
    image=image.add_local_file(root/"models/passenger-feature/referent.json","/referent.json")
    for name in ("passenger_place_modal","passenger_single_vehicle_call","passenger_frontage_call"):
        image=image.add_local_file(root/"tools"/(name+".py"),"/root/tools/"+name+".py")


@app.cls(image=image,gpu="A100-40GB",cpu=4,memory=30000,timeout=125,startup_timeout=300,
         min_containers=0,max_containers=1,scaledown_window=30,retries=0,
         volumes={"/base":modal.Volume.from_name("trackrag-qwen8b-public-weights").read_only(),
                  "/parking":modal.Volume.from_name("trackrag-parking-vistas-weights").read_only()},
         secrets=[modal.Secret.from_name("trackrag-parking-receipt-v1")])
class SingleVehicleReader(TrainedPlaceReader):
    @modal.enter()
    def load(self):
        from trackrag.dropoff.single_vehicle import RANKER
        super().load()
        coefficients=Path("/referent.json").read_bytes()
        if sha256(coefficients).hexdigest()!=RANKER:raise ValueError("Unexpected archived referent checkpoint")
        self.ranker=base64.b64encode(zlib.compress(coefficients)).decode("ascii")

    @modal.method()
    def observe(self,ctx,key,photo):
        from PIL import Image
        from trackrag.dropoff.contracts import validate_box
        from trackrag.dropoff.parking_evidence import MAX_IMAGE_BYTES,MAX_IMAGE_PIXELS,map_shape,signed_bytes
        from trackrag.dropoff.passenger import digest
        from trackrag.dropoff import single_vehicle as vehicle
        if (not isinstance(ctx,dict) or set(ctx)!={"recording_id","frame_id","size","image_sha256","tracks"}
                or not isinstance(photo,bytes) or not 0<len(photo)<=MAX_IMAGE_BYTES or sha256(photo).hexdigest()!=ctx["image_sha256"]
                or not isinstance(ctx["recording_id"],str) or not re.fullmatch(r"kartaview_\d+",ctx["recording_id"])
                or type(ctx["frame_id"]) is not int or ctx["frame_id"]<0
                or not isinstance(ctx["tracks"],list) or len(ctx["tracks"])>512
                or key!=digest({"recipe":vehicle.RECIPE,"context":ctx})):
            raise ValueError("Invalid public vehicle image or context")
        with Image.open(BytesIO(photo)) as p:
            if list(p.size)!=ctx["size"] or p.width*p.height>MAX_IMAGE_PIXELS:raise ValueError("Invalid source dimensions")
            original=p.convert("RGB")
        ids=[]
        for rid,cls,box in ctx["tracks"]:
            if type(rid) is not int or not isinstance(cls,str) or len(cls)>100:raise ValueError("Invalid object")
            validate_box(box,*ctx["size"]);ids.append(rid)
        if len(ids)!=len(set(ids)) or ids!=sorted(ids):raise ValueError("Duplicate or unordered observations")
        classes,_=self.segmenter.predict(original,map_shape({"width":original.width,"height":original.height}))
        options=vehicle.opportunities(ctx,classes)
        calls=vehicle.observation_calls(original,options,self.read)
        buf=BytesIO();Image.fromarray(classes).save(buf,format="PNG");encoded=buf.getvalue()
        body={"recipe":vehicle.RECIPE,"models":vehicle.MODELS,"image_sha256":ctx["image_sha256"],"context_sha256":digest(ctx),
              "class_map":{"png":base64.b64encode(encoded).decode("ascii"),"sha256":sha256(encoded).hexdigest()},
              "ranker_zlib":self.ranker,"calls":calls}
        return {"body":body,"signature":self.key.sign(signed_bytes(body)).hex()}
