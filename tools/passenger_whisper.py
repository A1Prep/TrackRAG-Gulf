"""Download the exact optional open-weight English voice model, or transcribe one local WAV."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from hashlib import file_digest
import json
import os
from pathlib import Path
import time

from trackrag.whisper_speech import FILES, MODEL_CONTENT_SHA256, MODEL_ID, REVISION, WhisperSpeechRecognizer, verify_model


def fetch(folder):
    import requests
    folder.mkdir(parents=True, exist_ok=True)
    source = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/"
    for name, (size, digest) in FILES.items():
        target = folder / name
        if target.exists():
            with target.open("rb") as stream:
                if target.stat().st_size == size and file_digest(stream, "sha256").hexdigest() == digest:
                    continue
            raise ValueError(f"An existing model file differs from the official revision: {name}")
        partial = folder / (name + ".partial")
        offset = partial.stat().st_size if partial.is_file() else 0
        if offset > size:
            raise ValueError(f"Partial model file exceeds its official byte count: {name}")
        url = source + name
        def download_chunk(lo):
            hi = min(lo + 4 * 1024**2, size) - 1
            for attempt in range(3):
                try:
                    response = requests.get(url, params={"download":"true","trackrag_span":f"{lo}-{hi}"},
                            headers={"Range":f"bytes={lo}-{hi}"}, timeout=(7,45))
                    response.raise_for_status()
                    whole = lo == 0 and hi + 1 == size and response.status_code == 200
                    ranged = response.status_code == 206 and response.headers.get("Content-Range") == f"bytes {lo}-{hi}/{size}"
                    if not (whole or ranged) or len(response.content) != hi - lo + 1:
                        raise ValueError("Hugging Face returned a different byte range")
                    return response.content
                except requests.RequestException:
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
        starts = list(range(offset, size, 4 * 1024**2))
        with ThreadPoolExecutor(max_workers=4) as workers, partial.open("ab") as stream:
            for i in range(0, len(starts), 4):
                for block in workers.map(download_chunk, starts[i:i+4]):
                    stream.write(block)
                    stream.flush()
        with partial.open("rb") as stream:
            if file_digest(stream, "sha256").hexdigest() != digest:
                raise ValueError(f"Downloaded model file did not match the official hash: {name}")
        os.replace(partial, target)
    verify_model(folder)
    return {"model_id":MODEL_ID, "revision":REVISION, "source":source, "bytes":sum(size for size,_ in FILES.values()),
            "model_content_sha256":MODEL_CONTENT_SHA256, "directory":str(folder.resolve())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    download = actions.add_parser("fetch")
    download.add_argument("--out", type=Path, required=True)
    run = actions.add_parser("transcribe")
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--wav", type=Path, required=True)
    args = parser.parse_args()
    result = fetch(args.out) if args.action == "fetch" else asdict(WhisperSpeechRecognizer(args.model_dir).transcribe_wav(args.wav.read_bytes()))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
