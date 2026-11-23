"""Run a bounded point-or-abstain test on the official Qwen3.6 35B FP8 checkpoint."""
from pathlib import Path
import modal

from tools import passenger_qwen_point_protocol as protocol
from tools.passenger_qwen35_download import MODEL_FOLDER, REPO, REVISION, VOLUME


app = modal.App('trackrag-public-qwen35-point-trial')
IMAGE = 'vllm/vllm-openai:v0.19.1@sha256:2622f38a0aa646c15ccc27bd5033911a58fd94ac69fd8f86aba0692d77cfe5b9'
image = modal.Image.from_registry(IMAGE, add_python="3.11").entrypoint([])
if modal.is_local():
    image = protocol.mount_trial(image).add_local_file(Path(__file__).with_name('passenger_qwen35_download.py'),
                                '/root/tools/passenger_qwen35_download.py')
    image = image.add_local_file(Path(__file__).with_name('passenger_qwen35_native.py'),
                                 '/root/tools/passenger_qwen35_native.py')


@app.function(image=image, gpu='H100', cpu=6, memory=56000, timeout=570, retries=0, max_containers=1,
              single_use_containers=True, volumes={'/weights':modal.Volume.from_name(VOLUME).read_only()})
def run_points(cases, task='point'):
    import json
    import os
    import pickle
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as work:
        source = Path(work) / 'inputs.pickle'
        destination = Path(work) / 'results.json'
        with source.open('wb') as handle:
            pickle.dump((cases, (REPO, REVISION, MODEL_FOLDER, IMAGE), task), handle)
        env = dict(os.environ, PYTHONPATH='/root', PYTHONNOUSERSITE='1')
        # vLLM's official container puts its CUDA wheels in Ubuntu's native Python 3.12.
        # Modal's separately pinned interpreter owns only the authenticated transport.
        subprocess.run(['/usr/bin/python3.12','-u','/root/tools/passenger_qwen35_native.py',
                        str(source),str(destination)],check=True,env=env,timeout=535)
        return json.loads(destination.read_bytes())


@app.local_entrypoint()
def main(manifest: str, out: str, task: str = 'point'):
    if task not in {'point', 'sign'}:
        raise ValueError('Choose point for parking or sign for the two-photo visual control')
    protocol.run_local(run_points, manifest, out, task=task)
