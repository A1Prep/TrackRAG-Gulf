"""Compute text-conditioned ground and landmark evidence on listed public photos."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import modal

REVISION = "999e0328d9e10b484360c477313983f9afdd7050"
QUERIES = (
    "an empty parking bay for a car", "a driveway, vehicle entrance or paved pull-in",
    "the lane of traffic directly ahead of the camera", "a raised pedestrian sidewalk",
    "a parked car or truck", "a curb or edge of the road", "a gate, fence, chain or barrier",
)
app = modal.App("trackrag-frontage-language-masks")
cache = modal.Volume.from_name("trackrag-frontage-models")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.6.0","torchvision==0.21.0","numpy==1.26.4","Pillow==11.1.0")
         .pip_install("transformers==4.57.1","scipy==1.15.2").env({"HF_HUB_OFFLINE":"1"}))


def landmark(words):
    reference = re.split(r"\b(?:beside|near|in front of|by|at|along|on|in|past|before)\s+(?=(?:that|the|those|this|an?)\b)",
                         words, maxsplit=1, flags=re.I)[-1]
    return re.sub(r"\s+on (?:my|our|the|your) (?:left|right)\b.*$","",reference.strip(" ?.!"),flags=re.I)


@app.cls(image=image,gpu="L4",cpu=2,memory=8500,timeout=100,min_containers=0,max_containers=1,scaledown_window=20,
         volumes={"/weights":cache.read_only()})
class Surfaces:
    @modal.enter()
    def load(self):
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
        path="/weights/"+REVISION
        self.proc=CLIPSegProcessor.from_pretrained(path,local_files_only=True)
        self.model=CLIPSegForImageSegmentation.from_pretrained(path,local_files_only=True,use_safetensors=True).to("cuda").eval()

    @modal.method()
    def compute(self, data, key, words):
        from PIL import Image
        import numpy as np
        import torch
        if len(data)>42*1024*1024 or sha256(data).hexdigest()!=key or not 1<=len(words)<=1600:
            raise ValueError("Invalid public image or words")
        with Image.open(BytesIO(data)) as src:
            photo=src.convert("RGB")
        queries=[*QUERIES,landmark(words)]
        # Preserve the full image, then add overlapping square-ish crops so small
        # street signs and pavement on either side also receive model resolution.
        w,h=photo.size
        target=(144,96)
        planes=np.zeros((len(queries),target[1],target[0]),dtype=np.float32)
        weight=np.zeros((target[1],target[0]),dtype=np.float32)
        for x0,x1 in [(0,144),(0,94),(50,144)]:
            crop=photo.crop((round(w*x0/144),0,round(w*x1/144),h))
            inputs=self.proc(text=queries,images=[crop]*len(queries),return_tensors="pt",padding=True).to("cuda")
            with torch.inference_mode(): logits=self.model(**inputs).logits.float().cpu().numpy()
            # Blend overlapping fields away from crop boundaries in the original view.
            blend=np.ones((1,x1-x0),dtype=np.float32)
            if x0: blend[0,:15]=np.linspace(.1,1,15)
            if x1<144: blend[0,-15:]=np.linspace(1,.1,15)
            for i,channel in enumerate(logits):
                planes[i,:,x0:x1]+=np.asarray(Image.fromarray(channel).resize((x1-x0,target[1]),Image.Resampling.BILINEAR))*blend
            weight[:,x0:x1]+=blend
        out=BytesIO()
        np.savez_compressed(out,logits=(planes/weight).astype("float16"),revision=REVISION,source=key,words=words,queries=queries)
        return out.getvalue()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("sources",type=Path);p.add_argument("rows",type=Path);p.add_argument("images",type=Path);p.add_argument("out",type=Path)
    a=p.parse_args()
    allow={r["sha256"] for r in json.loads(a.sources.read_bytes())["photos"]
           if re.fullmatch(r"https://kartaview\.org/details/[1-9]\d*/\d+/?",r["photo_url"])}
    a.out.mkdir(parents=True,exist_ok=True)
    with modal.enable_output(),app.run():
        worker=Surfaces()
        for line in a.rows.read_bytes().splitlines():
            r=json.loads(line);key=r["sha256"];rid=r["id"]
            if not re.fullmatch(r"[a-zA-Z0-9_-]+",rid) or key not in allow:raise ValueError("Unexpected trial input")
            path=a.images/(key+".jpg");data=path.read_bytes()
            if sha256(data).hexdigest()!=key:raise ValueError("Original photo changed")
            packet=worker.compute.remote(data,key,r["request"])
            (a.out/(rid+".npz")).write_bytes(packet)
            print(rid,len(packet),flush=True)

if __name__=="__main__":
    main()
