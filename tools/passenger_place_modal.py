"""Deploy the private trained-8B place verifier for explicitly selected public photographs."""
import base64
from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path
import re

import modal

from tools.passenger_place_call import APP, PROCESSOR_REVISION

app = modal.App(APP)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.6.0","torchvision==0.21.0","Pillow==11.1.0","numpy==1.26.4")
         .pip_install("transformers==5.1.0","peft==0.18.1","accelerate==1.12.0","bitsandbytes==0.49.1",
                      "safetensors==0.8.0","scipy==1.15.2","cryptography==50.0.2",
                      "opencv-python-headless==4.10.0.84")
         .env({"HF_HUB_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false","HF_HUB_DISABLE_PROGRESS_BARS":"1",
               "HF_HUB_DISABLE_TELEMETRY":"1","OMP_NUM_THREADS":"4"}))
if modal.is_local():
    root = Path(__file__).resolve().parents[1]
    processor = root/"models/dropoff-processor-cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots"/PROCESSOR_REVISION
    image = (image.add_local_dir(processor,"/processor")
             .add_local_dir(root/"models/passenger-feature/adapter","/adapter")
             .add_local_dir(root/"trackrag","/root/trackrag",ignore=lambda p:p.is_file() and p.suffix != ".py"))
    for name in ("dropoff_predict","passenger_parking_study","passenger_parking_anchors","passenger_parking_candidates",
                 "passenger_road_points","passenger_place_call"):
        image = image.add_local_file(root/"tools"/(name+".py"),"/root/tools/"+name+".py")


def fingerprints(path, names):
    from hashlib import file_digest
    from trackrag.dropoff.passenger import digest
    hashes = {}
    for name in names:
        with (path/name).open("rb") as source:
            hashes[name] = file_digest(source, "sha256").hexdigest()
    return digest(hashes)


@app.cls(image=image,gpu="A100-40GB",cpu=4,memory=30000,timeout=200,startup_timeout=360,
         min_containers=0,max_containers=1,scaledown_window=30,retries=0,
         volumes={"/base":modal.Volume.from_name("trackrag-qwen8b-public-weights").read_only(),
                  "/parking":modal.Volume.from_name("trackrag-parking-vistas-weights").read_only()},
         secrets=[modal.Secret.from_name("trackrag-parking-receipt-v1")])
class PassengerPlaces:
    @modal.enter()
    def load(self):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from tools.passenger_parking_study import ParkingSegmenter, SEG_REVISION
        self.key = Ed25519PrivateKey.from_private_bytes(base64.b64decode(os.environ.pop("PARKING_SIGNING_KEY"),validate=True))
        self.segmenter = ParkingSegmenter(Path("/parking")/SEG_REVISION)
        self.qwen = None

    def load_language_model(self):
        import torch
        from peft import PeftModel
        from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
        from trackrag.dropoff.place_evidence import FINGERPRINTS
        from trackrag.dropoff.place_grounding import BASE

        base = Path("/base")/BASE
        names = ("model-00001-of-00002.safetensors","model-00002-of-00002.safetensors","config.json","model.safetensors.index.json")
        actual = {"base_files":fingerprints(base,names)}
        for key,path in (("adapter_files",Path("/adapter")),("processor_files",Path("/processor"))):
            actual[key] = fingerprints(path, (p.name for p in path.iterdir() if p.is_file()))
        if actual != FINGERPRINTS:
            raise ValueError("Trained adapter, base model or processor differs from the verified release")
        self.processor = AutoProcessor.from_pretrained("/processor",local_files_only=True,trust_remote_code=False,
                                                      min_pixels=65536,max_pixels=262144)
        config = AutoConfig.from_pretrained(base,local_files_only=True,trust_remote_code=False)
        if "model.visual" not in config.quantization_config["llm_int8_skip_modules"]:
            config.quantization_config["llm_int8_skip_modules"].append("model.visual")
        model = Qwen3VLForConditionalGeneration.from_pretrained(base,config=config,device_map={"":"cuda:0"},dtype=torch.bfloat16,
                    attn_implementation="sdpa",local_files_only=True,trust_remote_code=False)
        trained = PeftModel.from_pretrained(model,"/adapter",is_trainable=False,local_files_only=True).eval()
        if any(param.device.type != "cuda" for param in trained.parameters()):
            raise ValueError("The trained place model must execute on the GPU")
        self.qwen = trained

    def read(self, guide, words, pixels):
        from PIL import Image
        import torch
        if self.qwen is None:
            self.load_language_model()
        with Image.open(BytesIO(pixels)) as source:
            photo = source.convert("RGB")
        messages = [{"role":"system","content":guide},{"role":"user","content":[{"type":"image","image":photo},
                                                                                {"type":"text","text":words}]}]
        prompt = self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        data = self.processor(text=[prompt],images=[photo],return_tensors="pt",truncation=False)
        data.pop("token_type_ids",None)
        tokens = data["input_ids"].shape[-1]
        if tokens > 3072:
            raise ValueError("Place instruction exceeds model context")
        with torch.inference_mode():
            result = self.qwen.generate(**data.to("cuda:0"),do_sample=False,use_cache=True,max_new_tokens=16,logits_to_keep=1)
        return self.processor.batch_decode(result[:,tokens:],skip_special_tokens=True)[0].strip().upper()

    @modal.method()
    def interpret(self, context, input_sha, photo):
        from PIL import Image
        from trackrag.dropoff import place_grounding as place
        from trackrag.dropoff.parking_evidence import MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS, map_shape, signed_bytes
        from trackrag.dropoff.passenger import digest
        from trackrag.dropoff.place_evidence import MODELS
        from trackrag.dropoff.contracts import validate_box

        if (not isinstance(photo,bytes) or len(photo)>MAX_IMAGE_BYTES or not re.fullmatch("[a-f0-9]{64}", input_sha)
                or set(context) != {"words","size","current_boxes","image_sha256"}
                or sha256(photo).hexdigest() != context["image_sha256"]
                or not isinstance(context["words"],str) or not 1 <= len(context["words"]) <= 1600
                or not isinstance(context["current_boxes"],list) or len(context["current_boxes"])>3000):
            raise ValueError("Invalid place-model input")
        with Image.open(BytesIO(photo)) as source:
            if list(source.size) != context["size"] or source.width*source.height > MAX_IMAGE_PIXELS:
                raise ValueError("Invalid place image dimensions")
            for box in context["current_boxes"]:
                validate_box(box,*source.size)
            original = source.convert("RGB")
        classes, _ = self.segmenter.predict(original,map_shape({"width":original.width,"height":original.height}))
        calls = []

        def infer(kind, guide, pixels):
            response = self.read(guide, context["words"], pixels)
            calls.append({**place.call_identity(kind, guide, context["words"], pixels),"response":response})
            return response

        place.apply_steps(context, classes, original, infer)
        memory = BytesIO()
        Image.fromarray(classes).save(memory, format="PNG")
        packed = memory.getvalue()
        body = {"recipe":place.METHOD,"models":MODELS,"input_sha256":input_sha,"context_sha256":digest(context),
                "image_sha256":context["image_sha256"],"calls":calls,
                "class_map":{"sha256":sha256(packed).hexdigest(),"png":base64.b64encode(packed).decode("ascii")}}
        return {"body":body,"signature":self.key.sign(signed_bytes(body)).hex()}
