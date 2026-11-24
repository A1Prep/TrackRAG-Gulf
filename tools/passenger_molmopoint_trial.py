"""Run the predeclared MolmoPoint development study on allowlisted original KartaView images."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from time import monotonic

import modal

from tools.passenger_frontage_pointing import runtime, weights
from tools.passenger_release import PUBLIC_PHOTO_LISTS, ROOT


REVISION = "188130f961c8e0888a34e11121a1423c461a01ba"
PLAN = ROOT / "experiments/dropoff/passenger-molmopoint-development-plan-20261124.json"
PLAN_SHA = "de50aac891cb940902f3103caf8ab26e01097d2ccf3619a4236fabbf11018427"
app = modal.App("trackrag-public-molmopoint-development")


def checked_inputs(local_sources):
    raw = PLAN.read_bytes()
    canonical = raw.replace(b"\r\n",b"\n").replace(b"\n",b"\r\n")
    if sha256(canonical).hexdigest()!=PLAN_SHA:
        raise ValueError("The pointing experiment plan changed")
    plan=json.loads(raw)
    local=json.loads(Path(local_sources).read_bytes())
    allowed = {r["sha256"]:r["photo_url"].rstrip("/") for name in PUBLIC_PHOTO_LISTS
               for r in json.loads((ROOT/"experiments/dropoff"/name).read_bytes())["photos"]}
    originals={}
    turns=[]
    for case in plan["cases"]:
        key=case["photo_sha256"]
        if allowed.get(key)!=case["photo_url"].rstrip("/"):
            raise ValueError("The original is outside the existing public KartaView source list")
        if key not in originals:
            path=Path(local[case["id"].split("-")[0]]["path"])
            data=path.read_bytes()
            if not 0<len(data)<42*1024*1024 or sha256(data).hexdigest()!=key:
                raise ValueError("Original photograph bytes changed")
            originals[key]=data
        prompt=plan["ground_prompt"].format(words=case["text"]) if case["task"]=="ground" else case["text"]
        # Ground-truth labels, reference polygons and local paths never enter the point generator.
        turns.append((case["id"],key,prompt))
    return originals,turns


@app.function(image=runtime,gpu="A100-40GB",cpu=4,memory=48000,timeout=430,retries=0,
              volumes={"/weights":weights.read_only()})
def run_batch(photos,turns):
    import numpy as np
    from PIL import Image
    import torch
    from transformers import AutoModelForImageTextToText,AutoProcessor

    started=monotonic()
    model_path="/weights/"+REVISION
    processor=AutoProcessor.from_pretrained(model_path,trust_remote_code=True,local_files_only=True,padding_side="left")
    model=AutoModelForImageTextToText.from_pretrained(model_path,trust_remote_code=True,local_files_only=True,
                                                     dtype=torch.bfloat16,use_safetensors=True,device_map={"":"cuda:0"}).eval()
    torch.cuda.synchronize()
    loaded=monotonic()-started
    output=[]
    for case_id,key,prompt in turns:
        tik=monotonic()
        try:
            data=photos[key]
            if sha256(data).hexdigest()!=key:
                raise ValueError("Original photograph changed in transit")
            with Image.open(BytesIO(data)) as original:
                if original.format!="JPEG" or original.width*original.height>44_000_000:
                    raise ValueError("Unexpected photo type or size")
                image=original.convert("RGB")
            messages=[{"role":"user","content":[{"type":"text","text":prompt},{"type":"image","image":image}]}]
            inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                return_tensors="pt",return_dict=True,padding=True,return_pointing_metadata=True)
            meta=inputs.pop("metadata")
            inputs={key:value.to("cuda:0") for key,value in inputs.items()}
            with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
                seq=model.generate(**inputs,logits_processor=model.build_logit_processor_from_inputs(inputs),
                                   do_sample=False,max_new_tokens=96)
            tokens=seq[:,inputs["input_ids"].size(1):]
            text=processor.post_process_image_text_to_text(tokens,skip_special_tokens=False,
                                                         clean_up_tokenization_spaces=False)[0]
            raw=model.extract_image_points(text,meta["token_pooling"],meta["subpatch_mapping"],meta["image_sizes"])
            image_sizes=np.asarray(meta["image_sizes"]).reshape(-1,2).tolist()
            if image_sizes!=[list(image.size)]:
                raise ValueError("Point decoder disagrees with the original photo dimensions")
            points=[]
            for object_id,image_index,x,y in raw:
                if int(image_index)!=0 or not np.isfinite([x,y]).all() or not(0<=x<image.width and 0<=y<image.height):
                    raise ValueError("Point is outside this original image")
                points.append({"object_id":int(object_id),"image":int(image_index),"xy":[float(x),float(y)]})
            torch.cuda.synchronize()
            row={"id":case_id,"photo_sha256":key,"size":list(image.size),"prompt_sha256":sha256(prompt.encode()).hexdigest(),
                 "tokens":int(tokens.size(1)),"text":text,"points":points}
        except Exception as exc:
            row={"id":case_id,"photo_sha256":key,"error":type(exc).__name__+": "+str(exc)[:550]}
        row["inference_s"]=round(monotonic()-tik,3)
        output.append(row)
        print(case_id,"error" if row.get("error") else str(len(row["points"]))+" points",row["inference_s"],flush=True)
        if len(output)==1 and row.get("error"):
            # Detect an incompatible execution environment before repeating a failing model call.
            raise RuntimeError("Pointing smoke test failed: "+row["error"])
    return {"repository":"allenai/MolmoPoint-8B","revision":REVISION,"device":torch.cuda.get_device_name(),
            "dtype":"bfloat16","weights_load_s":round(loaded,3),"total_remote_s":round(monotonic()-started,3),
            "peak_allocated_gpu_gib":round(torch.cuda.max_memory_allocated()/1024**3,3),"trials":output}


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--local-sources",type=Path,required=True)
    cli.add_argument("--out",type=Path,required=True)
    args=cli.parse_args()
    if args.out.exists():
        raise ValueError("Choose a new output path to preserve past trials")
    originals,turns=checked_inputs(args.local_sources)
    start=monotonic()
    with modal.enable_output(),app.run():
        received=run_batch.remote(originals,turns)
    received["local_elapsed_s"]=round(monotonic()-start,3)
    received["plan_sha256"]=PLAN_SHA
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(received,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("Completed",len(received["trials"]),"trials on",received["device"],flush=True)


if __name__=="__main__":
    main()
