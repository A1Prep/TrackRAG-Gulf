"""Native Python worker for the official vLLM CUDA container."""
from pathlib import Path
from tools import passenger_qwen_point_protocol as protocol


SIGN_QUESTION = 'What business name can you read on the most prominent storefront sign? Reply with the visible words or NONE.'


def run(cases, identity, task='point'):
    import json
    import os
    from time import perf_counter

    os.environ.update(HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false',
                      VLLM_DEEP_GEMM_WARMUP='skip',
                      VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1', VLLM_WORKER_MULTIPROC_METHOD='spawn')
    import torch
    import vllm
    import transformers
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    REPO, REVISION, MODEL_FOLDER, IMAGE = identity
    images = list(protocol.checked_inputs(cases))
    if task not in {'point', 'sign'}:
        raise ValueError('Unknown trial task')
    if task == 'sign' and (len(images) != 2 or any(item['words'] != SIGN_QUESTION for item, _ in images)):
        raise ValueError('The vision control uses precisely two photos and one neutral sign-reading question')
    base = Path('/weights') / MODEL_FOLDER
    record = json.loads((base/'trial-download-verification.json').read_bytes())
    if (record['repository'],record['revision'],record['shards'],record['shard_bytes']) != (REPO,REVISION,42,37463662160):
        raise ValueError('Cached checkpoint differs from the pinned official model')
    print(f'Loading {REPO} {REVISION} on {torch.cuda.get_device_name(0)}', flush=True)
    start = perf_counter()
    processor = AutoProcessor.from_pretrained(str(base), local_files_only=True, trust_remote_code=False)
    llm = LLM(model=str(base), tensor_parallel_size=1, max_model_len=4096, max_num_seqs=1,
              gpu_memory_utilization=.84, enforce_eager=True, limit_mm_per_prompt={'image':1},
              seed=1, trust_remote_code=False, generation_config='vllm', enable_prefix_caching=False)
    init_seconds = perf_counter()-start
    params = SamplingParams(temperature=0.0, max_tokens=protocol.MAX_NEW_TOKENS, seed=1)
    results=[]
    for item, photo in images:
        dialogue = protocol.messages(item['words'])
        if task == 'sign':
            dialogue = [{'role':'user','content':[{'type':'image'},{'type':'text','text':SIGN_QUESTION}]}]
        prompt = processor.apply_chat_template(dialogue, tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
        before = perf_counter()
        output, = llm.generate([{'prompt':prompt, 'multi_modal_data':{'image':photo}}], params, use_tqdm=False)
        generated, = output.outputs
        results.append(dict(item,raw=generated.text,output_tokens=len(generated.token_ids),
                     prompt_tokens=len(output.prompt_token_ids),seconds=round(perf_counter()-before,3)))
        print(f"Completed case {len(results)}/{len(images)} in {results[-1]['seconds']} s",flush=True)
    return {'model':REPO,'revision':REVISION,'loader':'vllm','vllm':vllm.__version__, 'image':IMAGE,
            'torch':str(torch.__version__),'transformers':transformers.__version__, 'quantization':'official fine-grained FP8',
            'gpu':torch.cuda.get_device_name(0),'model_load_seconds':round(init_seconds,2),'max_new_tokens':protocol.MAX_NEW_TOKENS,
            'temperature':0,'thinking':False,'results':results}


if __name__ == '__main__':
    import pickle
    import json
    import sys

    request, destination = map(Path, sys.argv[1:])
    with request.open('rb') as handle:
        arguments = pickle.load(handle)
    result = run(*arguments)
    destination.write_text(json.dumps(result), encoding='utf-8')
