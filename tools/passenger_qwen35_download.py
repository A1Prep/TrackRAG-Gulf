"""Fetch the official spatial-grounding trial checkpoint directly into the private Modal volume."""
from pathlib import Path
import modal

REPO = "Qwen/Qwen3.6-35B-A3B-FP8"
REVISION = "c653bba46dfa06e9658dcc0b0dc87fe4685f14a2"
VOLUME = "trackrag-qwen36-35b-fp8"
MODEL_FOLDER = "qwen36-35b-" + REVISION[:12]
app = modal.App("trackrag-public-qwen35-download")
volume = modal.Volume.from_name(VOLUME,create_if_missing=True)


@app.function(image=modal.Image.debian_slim(python_version="3.11"),cpu=4,memory=8192,timeout=900,
              volumes={"/weights":volume},retries=0)
def fetch():
    from concurrent.futures import ThreadPoolExecutor,as_completed
    import hashlib
    import json
    import os
    import time
    from urllib.parse import quote
    from urllib.request import Request,urlopen

    dest=Path("/weights")/MODEL_FOLDER
    dest.mkdir(exist_ok=True)
    with urlopen(f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true",timeout=30) as source:
        info=json.load(source)
    if info["sha"]!=REVISION:
        raise ValueError("Unexpected official revision")
    files=[f for f in info["siblings"] if "/" not in f["rfilename"] and not f["rfilename"].startswith(".")]
    start=time.perf_counter()
    def file_hash(path):
        with path.open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()
    def download(f):
        name=f["rfilename"];size=f["size"];expected=f.get("lfs",{}).get("sha256")
        target=dest/name
        if target.is_file() and target.stat().st_size==size:
            digest=file_hash(target)
            if expected is None or digest==expected:return {"name":name,"bytes":size,"sha256":digest}
        temporary=target.with_suffix(target.suffix+".partial")
        for attempt in range(3):
            try:
                address=f"https://huggingface.co/{REPO}/resolve/{REVISION}/{quote(name)}?download=true&fetch={int(time.time())}"
                digest=hashlib.sha256()
                with urlopen(Request(address,headers={"User-Agent":"TrackRAG-Model-Trial/1.0"}),timeout=65) as response, temporary.open("wb") as dst:
                    while data:=response.read(4*1024**2):
                        dst.write(data);digest.update(data)
                if temporary.stat().st_size!=size or expected and digest.hexdigest()!=expected:
                    raise ValueError(f"File checksum or length changed: {name}")
                os.replace(temporary,target)
                return {"name":name,"bytes":size,"sha256":digest.hexdigest()}
            except Exception:
                if attempt==2:raise
                time.sleep(2+attempt)
    checks=[]
    print(f"Downloading {len(files)} source artifacts to {VOLUME} at the official commit {REVISION}",flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(download,f) for f in files]
        for future in as_completed(futures):
            checks.append(future.result())
            if len(checks)%6==0:print(f"Checked {len(checks)}/{len(files)} ({sum(v['bytes'] for v in checks)/1e9:.2f} GB)",flush=True)
    shards=[r for r in checks if r["name"].endswith(".safetensors")]
    result={"repository":REPO,"revision":REVISION,"license":info["cardData"]["license"],"volume":VOLUME,
            "folder":MODEL_FOLDER,"files":sorted(checks,key=lambda row:row["name"]),"shards":len(shards),
            "shard_bytes":sum(row["bytes"] for row in shards),"seconds":round(time.perf_counter()-start,2)}
    (dest/"trial-download-verification.json").write_text(json.dumps(result,indent=2))
    volume.commit()
    return result


@app.local_entrypoint()
def main(out: str):
    import json
    result=fetch.remote()
    path=Path(out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2))
    print({k:v for k,v in result.items() if k!="files"},flush=True)
