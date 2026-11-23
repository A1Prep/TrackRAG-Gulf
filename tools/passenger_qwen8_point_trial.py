"""Paired direct-point control using the existing fingerprinted Qwen3-VL-8B fine-tune."""
from pathlib import Path
import modal

from tools import passenger_qwen_point_protocol as protocol
from tools.passenger_place_modal import TrainedPlaceReader, image


app = modal.App('trackrag-public-qwen8-point-control')
if modal.is_local():
    image = protocol.mount_trial(image).add_local_file(Path(__file__).with_name('passenger_place_modal.py'),
                                         '/root/tools/passenger_place_modal.py')


@app.function(image=image, gpu='A100-40GB', cpu=4, memory=30000, timeout=205, retries=0, max_containers=1,
              single_use_containers=True, volumes={'/base':modal.Volume.from_name('trackrag-qwen8b-public-weights').read_only()})
def run_points(cases):
    from time import perf_counter
    import torch
    import transformers
    from transformers import AutoProcessor
    from trackrag.dropoff.place_evidence import FINGERPRINTS
    from trackrag.dropoff.place_grounding import BASE
    from tools.passenger_place_call import PROCESSOR_REVISION

    images=list(protocol.checked_inputs(cases))
    start=perf_counter()
    reader=TrainedPlaceReader()
    reader.load_language_model()
    processor = AutoProcessor.from_pretrained('/processor',local_files_only=True,trust_remote_code=False,
                                            min_pixels=65536, max_pixels=protocol.MAX_PIXELS)
    init_seconds=perf_counter()-start
    results=[]
    for item,photo in images:
        template=processor.apply_chat_template(protocol.messages(item['words'],image_marker=photo),tokenize=False,
                                               add_generation_prompt=True,enable_thinking=False)
        data=processor(text=[template],images=[photo],return_tensors='pt',truncation=False)
        data.pop('token_type_ids',None)
        tokens=data['input_ids'].shape[-1]
        grid=data['image_grid_thw'].tolist()
        before=perf_counter()
        with torch.inference_mode():
            out=reader.qwen.generate(**data.to('cuda:0'),do_sample=False,use_cache=True,
                                     max_new_tokens=protocol.MAX_NEW_TOKENS,logits_to_keep=1)
        new=out[:,tokens:]
        raw=processor.batch_decode(new,skip_special_tokens=True)[0]
        results.append(dict(item,raw=raw,output_tokens=new.shape[-1],prompt_tokens=tokens,
                            image_grid=grid,seconds=round(perf_counter()-before,3)))
        print(f"Completed case {len(results)}/{len(images)} in {results[-1]['seconds']} s",flush=True)
    return {'model':'Qwen/Qwen3-VL-8B-Instruct + released Task 5 adapter','base_revision':BASE,
            'processor_revision':PROCESSOR_REVISION,'fingerprints':FINGERPRINTS,'quantization':'release base NF4 + LoRA adapter',
            'transformers':transformers.__version__, 'torch':str(torch.__version__), 'gpu':torch.cuda.get_device_name(0),
            'model_load_seconds':round(init_seconds,2),'max_new_tokens':protocol.MAX_NEW_TOKENS,
            'temperature':0,'thinking':False,'results':results}


@app.local_entrypoint()
def main(manifest: str, out: str):
    protocol.run_local(run_points, manifest, out)
