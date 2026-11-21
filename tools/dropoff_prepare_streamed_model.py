"""Prepare only the pinned 8B checkpoint and original pinned processor assets."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

from trackrag.dropoff.model import BASE_MODEL_ID, BASE_REVISION
from trackrag.dropoff.streamed import (STREAMED_MODEL_ID, STREAMED_REVISION,
                                      WEIGHT_SHA256, checkpoint_identity)

WEIGHT_BYTES={
    'model-00001-of-00002.safetensors':4999093335,
    'model-00002-of-00002.safetensors':2227305901,
}


def prepare(model_dir,processor_cache,*,offline=False):
    model_dir=Path(model_dir).expanduser().resolve()
    processor_cache=Path(processor_cache).expanduser().resolve()
    model_dir.mkdir(parents=True,exist_ok=True)
    missing=sum(size for name,size in WEIGHT_BYTES.items()
                if not (model_dir/name).is_file() or (model_dir/name).stat().st_size!=size)
    if not offline:
        if shutil.disk_usage(model_dir).free < missing+600*1024**2:
            raise RuntimeError('Insufficient checkpoint disk headroom; allow roughly 8 GiB on a fresh model volume')
        os.environ.setdefault('HF_HUB_DISABLE_XET','1')
        from huggingface_hub import snapshot_download
        snapshot_download(STREAMED_MODEL_ID,revision=STREAMED_REVISION,local_dir=model_dir,max_workers=1,
                          allow_patterns=['*.json','*.txt','*.safetensors','*.jinja'])
        # Restrict the original repository to processor/tokenizer files, not its
        # much larger unquantized weights. The two identities stay explicit.
        snapshot_download(BASE_MODEL_ID,revision=BASE_REVISION,cache_dir=processor_cache,max_workers=1,
                          allow_patterns=['*.json','*.txt','*.jinja','*.model'])
    identity=checkpoint_identity(model_dir,verify_weights=True)
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(BASE_MODEL_ID,revision=BASE_REVISION,cache_dir=processor_cache,
        local_files_only=True,trust_remote_code=False,min_pixels=65536,max_pixels=262144)
    result={'schema_version':1,'checkpoint':identity,'processor_model_id':BASE_MODEL_ID,
        'processor_revision':BASE_REVISION,'processor_class':type(processor).__name__,
        'processor_cache':str(processor_cache),'offline':offline,'model_training_executed':False}
    (model_dir/'prepared-model.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path,default=Path('models/qwen3-vl-8b-local-nf4'))
    parser.add_argument('--processor-cache',type=Path,default=Path('models/dropoff-processor-cache/hub'))
    parser.add_argument('--offline',action='store_true',help='Verify already prepared files without network access')
    args=parser.parse_args(argv)
    print(json.dumps(prepare(args.model_dir,args.processor_cache,offline=args.offline),indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
