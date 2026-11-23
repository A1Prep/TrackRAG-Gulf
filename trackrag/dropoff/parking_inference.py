"""Run both real checkpoints for one current passenger request in a private GPU worker."""
import base64
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from tools.passenger_parking_study import ParkingSegmenter, QwenChooser, QWEN_REVISION, SEG_REVISION
from .parking_evidence import INSPECTED_RECIPE, REVIEWED_RECIPE, current_only, map_shape, portable_request, signed_bytes, source_input, view_digest
from .parking_gap_review import steps as inspected_steps
from .parking_reference_board import steps as reviewed_steps
from .parking_views import render
from .passenger_parking import parking_instruction


class ParkingModels:
    def __init__(self, parking_path, qwen_path, signing_key):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        self.key = Ed25519PrivateKey.from_private_bytes(base64.b64decode(signing_key, validate=True))
        self.segmenter = ParkingSegmenter(parking_path)
        self.chooser = QwenChooser(qwen_path)

    def run(self, wire_request, photo, recipe=None):
        recipe = REVIEWED_RECIPE if recipe is None else recipe
        if recipe not in (REVIEWED_RECIPE, INSPECTED_RECIPE):
            raise ValueError("Unsupported live parking recipe")
        request, image_sha, input_sha = source_input(wire_request, photo)
        request = current_only(portable_request(request))
        frame = request["scene"]["frame"]
        words = parking_instruction(request)
        if not words:
            raise ValueError("Only active two-vehicle instructions are accepted")
        with Image.open(BytesIO(photo)) as src:
            classes, _ = self.segmenter.predict(src.convert("RGB"),map_shape(frame))
        png = BytesIO()
        Image.fromarray(classes).save(png,format="PNG")
        data = png.getvalue()
        records = []
        with TemporaryDirectory() as directory:
            root=Path(directory)
            original=root/"original.frame"
            original.write_bytes(photo)
            machine=(inspected_steps if recipe == INSPECTED_RECIPE else reviewed_steps)(request,classes)
            try:
                step=next(machine)
                while True:
                    image=render(original,frame,step["view"],root/(step["kind"]+".jpg"))
                    rendered_sha=sha256(image.read_bytes()).hexdigest()
                    row={"id":request["request_id"], "request":words, "sha256":image_sha, "original":str(original),
                         "view":str(image), "view_sha":rendered_sha}
                    if step["kind"]=="anchors":
                        raw=self.chooser.infer_anchors([row])[0]
                    else:
                        # Board/inspection reads get more image tokens than the small marked-letter view.
                        small_board = step["view"]["type"] in ("points", "parking_components")
                        processor = self.chooser.processor if small_board else self.chooser.anchor_processor
                        raw=self.chooser._infer([row], processor=processor, thinking=False, limit=96,
                            system_for=lambda _:step["system"], image_key="view",hash_key="view_sha",maximum_batch=1,mode=step["kind"])[0]
                    system_sha=sha256(step["system"].encode()).hexdigest()
                    if raw["prompt_sha256"]!=system_sha or raw["sent_sha"]!=rendered_sha:
                        raise ValueError("Generated response is bound to a different model input")
                    answer=raw["decoded_final_answer"]
                    records.append({"kind":step["kind"],"system_sha256":system_sha,"view_sha256":view_digest(step["view"]),
                                    "rendered_sha256":rendered_sha,"response":answer,"chat_sha256":raw["rendered_chat_sha256"],
                                    "generated_tokens":raw["generated_tokens"],"seconds":raw["seconds_for_batch"]})
                    step=machine.send(answer)
            except StopIteration:
                pass
        body={"recipe":recipe,"models":[QWEN_REVISION,SEG_REVISION],"input_sha256":input_sha,"image_sha256":image_sha,
              "words":words,"class_map":{"sha256":sha256(data).hexdigest(),"png":base64.b64encode(data).decode("ascii")},
              "calls":records}
        return {"body":body,"signature":self.key.sign(signed_bytes(body)).hex()}
