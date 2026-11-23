"""Batched text observations from the pinned model in the native vLLM environment."""
import io
import json
import os
from pathlib import Path
import sys


def load_runtime(folder):
    os.environ.update(HF_HUB_OFFLINE="1",HF_HUB_DISABLE_TELEMETRY="1",VLLM_NO_USAGE_STATS="1",
                      VLLM_DEEP_GEMM_WARMUP="skip",VLLM_WORKER_MULTIPROC_METHOD="spawn",TOKENIZERS_PARALLELISM="false")
    from transformers import AutoProcessor
    from vllm import LLM
    processor=AutoProcessor.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    model=LLM(model=folder,tensor_parallel_size=1,max_model_len=6144,max_num_seqs=4,gpu_memory_utilization=.86,
              enforce_eager=True,limit_mm_per_prompt={"image":1},seed=1,trust_remote_code=False,
              generation_config="vllm",enable_prefix_caching=False)
    return processor,model


def generate(processor,model,image_rows,words,limit=145):
    import base64
    from PIL import Image
    from vllm import SamplingParams
    chat=processor.apply_chat_template([{"role":"user","content":[{"type":"image"},{"type":"text","text":words}]}],
                                      tokenize=False,add_generation_prompt=True,enable_thinking=False)
    params=SamplingParams(temperature=0,max_tokens=limit,seed=1)
    replies=[]
    for start in range(0,len(image_rows),6):
        batch=[]
        for data in image_rows[start:start+6]:
            with Image.open(io.BytesIO(base64.b64decode(data))) as im:
                batch.append({"prompt":chat,"multi_modal_data":{"image":im.convert("RGB")}})
        result=model.generate(batch,params,use_tqdm=False)
        replies.extend(out.outputs[0].text for out in result)
        print(f"Described {len(replies)}/{len(image_rows)} locations.",flush=True)
    return replies


def main(source,destination,folder,prompt):
    processor,model=load_runtime(folder)
    replies=generate(processor,model,json.loads(Path(source).read_bytes()),Path(prompt).read_text())
    Path(destination).write_text(json.dumps(replies))


if __name__=="__main__":
    main(*sys.argv[1:])
