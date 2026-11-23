"""Evaluate a frozen pointing specialist on explicitly listed public UAE photographs."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re

import modal

REVISION = "e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"
PROMPT = (
    "The passenger says: {words}\n"
    "Point to the middle of the vacant ground where the passenger wants the car to pull in. "
    "Choose a visible vehicle-sized parking space, pull-in, or driveway at the specified landmark. "
    "Do not point at the landmark itself, a sidewalk, the moving lane, or a parked car. "
    "If the requested area cannot be seen, is occupied, or is behind a barrier, respond NONE. "
    "Otherwise point to precisely one spot."
)
CHECK = (
    "The passenger says: {words}\n"
    "The small yellow ring marks one point proposed by an assistant. Inspect its centre and "
    "the surroundings. Is there an open vehicle-sized bay, pull-in or driveway at that exact "
    "point which matches the passenger's requested landmark and side? An ordinary travel lane, "
    "walkway, obstructed entrance or occupied stall does not qualify. Reply YES, NO or UNCLEAR."
)

app = modal.App("trackrag-public-frontage-pointing-trial")
weights = modal.Volume.from_name("trackrag-frontage-models")
runtime = (modal.Image.debian_slim(python_version="3.11")
           .pip_install("torch==2.6.0", "torchvision==0.21.0", "numpy==1.26.4", "Pillow==11.1.0")
           .pip_install("transformers==4.57.1", "accelerate==1.12.0", "einops==0.8.1", "scipy==1.15.2",
                        "opencv-python-headless==4.11.0.86", "decord==0.6.0", "timm==1.0.22")
           .env({"HF_HUB_OFFLINE":"1", "TOKENIZERS_PARALLELISM":"false"}))


def coordinates(answer):
    """Single-image Molmo2 point tags use thousandths along each original image axis."""
    tag = re.search(r'<points?\b[^>]*\bcoords="([^"]+)"', answer)
    if tag is None:
        return None
    numbers = [int(n) for n in re.findall(r"\d+", tag[1])]
    if len(numbers) not in (2, 3, 4) or not all(0 <= x <= 1000 for x in numbers[-2:]):
        return None
    return numbers[-2:]


def mark_center(image, point):
    from PIL import ImageDraw
    marked = image.copy()
    x, y = [min(d-1, round(v*d/1000)) for v,d in zip(point, marked.size)]
    radius = max(5, round(max(marked.size)*0.009))
    draw = ImageDraw.Draw(marked)
    for difference, color in ((2, "black"), (0,"yellow")):
        r = radius+difference
        draw.ellipse((x-r,y-r,x+r,y+r), outline=color, width=2)
    return marked


@app.cls(image=runtime, gpu="A100-40GB", cpu=4, memory=34000, timeout=180, startup_timeout=390,
         min_containers=0, max_containers=1, scaledown_window=35, volumes={"/weights":weights.read_only()})
class PointingTrial:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        base = "/weights/"+REVISION
        self.processor = AutoProcessor.from_pretrained(base, trust_remote_code=True, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            base, trust_remote_code=True, local_files_only=True, dtype=torch.bfloat16,
            use_safetensors=True, device_map={"":"cuda:0"}, attn_implementation="sdpa").eval()

    def read(self, image, words):
        import torch
        messages = [{"role":"user", "content":[{"type":"image","image":image},{"type":"text","text":words}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_tensors="pt", return_dict=True)
        inputs = {k:v.to("cuda:0") if hasattr(v,"to") else v for k,v in inputs.items()}
        with torch.inference_mode():
            generated = self.model.generate(**inputs, do_sample=False, max_new_tokens=90)
        return self.processor.tokenizer.decode(generated[0,inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    @modal.method()
    def predict(self, data, photo_sha, words):
        from PIL import Image
        if len(data)>42*1024*1024 or sha256(data).hexdigest()!=photo_sha or not 1 <= len(words) <= 1600:
            raise ValueError("Invalid photo or instruction")
        with Image.open(BytesIO(data)) as src:
            if src.width * src.height > 44_000_000:
                raise ValueError("Unexpected original photo dimensions")
            original=src.convert("RGB")
        original.thumbnail((1440,1050),Image.Resampling.LANCZOS)
        point_answer = self.read(original, PROMPT.format(words=words))
        point = coordinates(point_answer)
        check = self.read(mark_center(original,point), CHECK.format(words=words)) if point else None
        return {"image_sha256":photo_sha,"words":words,"revision":REVISION,"point_answer":point_answer,
                "point_1000":point,"check":check}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photos", required=True, type=Path, help="Public KartaView source manifest")
    parser.add_argument("--requests", required=True, type=Path, help="Neutral request records, one per line")
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    sources = {r["sha256"]:r for r in json.loads(args.photos.read_bytes())["photos"]
               if re.fullmatch(r"https://kartaview\.org/details/[1-9]\d*/\d+/?", r["photo_url"])}
    args.out.mkdir(parents=True, exist_ok=True)
    with modal.enable_output(), app.run():
        worker = PointingTrial()
        for raw in args.requests.read_bytes().splitlines():
            row = json.loads(raw)
            rid = row["id"]
            if not re.fullmatch(r"[a-zA-Z0-9_-]+", rid):
                raise ValueError("Invalid example ID")
            key = row["sha256"]
            if key not in sources:
                raise ValueError("Only allowlisted public photos may be used in this trial")
            data = (args.images/(key+".jpg")).read_bytes()
            if sha256(data).hexdigest()!=key:
                raise ValueError("Original photo hash mismatch")
            out = args.out/(rid+".json")
            if out.exists():
                prior = json.loads(out.read_bytes())
                if (prior["image_sha256"],prior["words"],prior["revision"]) != (key,row["request"],REVISION):
                    raise ValueError("Different request already occupies this result path")
                continue
            result = worker.predict.remote(data, key, row["request"])
            out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
            print(rid, result["point_1000"], result["check"], result["point_answer"][:110], flush=True)


if __name__ == "__main__":
    main()
