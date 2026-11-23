"""Cache pinned public pointing and segmentation weights in the research Modal volume."""
from pathlib import Path

import modal

from tools.passenger_frontage_pointing import REVISION as POINT_MODEL
from tools.passenger_frontage_surfaces import REVISION as SURFACE_MODEL

app = modal.App("trackrag-frontage-download")
cache = modal.Volume.from_name("trackrag-frontage-models", create_if_missing=True)
download_image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub==0.36.0")


@app.function(image=download_image, volumes={"/weights":cache}, timeout=750, retries=0, cpu=2, memory=4096)
def fetch(repo, revision):
    from huggingface_hub import snapshot_download

    path = Path("/weights") / revision
    snapshot_download(repo, revision=revision, local_dir=path, max_workers=5,
                      allow_patterns=["*.py","*.safetensors","*.json","*.txt","*.jinja","LICENSE"])
    cache.commit()
    return repo, sum(file.stat().st_size for file in path.glob("*.safetensors"))


if __name__ == "__main__":
    sources = [("allenai/Molmo2-8B",POINT_MODEL), ("CIDAS/clipseg-rd64-refined",SURFACE_MODEL)]
    with modal.enable_output(), app.run():
        for repo, size in fetch.starmap(sources, order_outputs=False):
            print(repo, size, "bytes", flush=True)
