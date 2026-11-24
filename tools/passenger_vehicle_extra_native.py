"""Checkpoint completed model batches so a time limit retains the original measured views."""
import base64
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import tempfile


def copy_verified_weights(folder, destination, records):
    """Read shards concurrently from the pinned volume, then load from ephemeral local disk."""
    def copy(record):
        relative = Path(record['name'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unexpected checkpoint path.')
        source = folder / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = sha256()
        length = 0
        with source.open('rb') as src, target.open('xb') as dst:
            while chunk := src.read(4 * 1024 * 1024):
                dst.write(chunk)
                digest.update(chunk)
                length += len(chunk)
        if length != record['bytes'] or digest.hexdigest() != record['sha256']:
            raise ValueError('A checkpoint file differs from the previously verified download.')

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(copy, records))


def measure(images, guide, output, generate_batch):
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        for start in range(0, len(images), 6):
            batch = images[start:start + 6]
            replies = generate_batch(batch, guide)
            for position, (data, reply) in enumerate(zip(batch, replies, strict=True), start):
                if not isinstance(reply, str) or len(reply) > 2500:
                    raise ValueError('Unexpected vehicle reader reply.')
                view_sha = sha256(base64.b64decode(data, validate=True)).hexdigest()
                stream.write(json.dumps([position, view_sha, reply]) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            print(f'Saved {start + len(batch)}/{len(images)} visual replies.', flush=True)


def completed_replies(path, expected_digests):
    """Recover consecutive complete records; an interrupted final write carries no measurement."""
    if not path.exists():
        return []
    replies = []
    with path.open('rb') as stream:
        for index, line in enumerate(stream):
            if not line.endswith(b'\n'):
                break
            position, digest, reply = json.loads(line)
            if (position != index or index >= len(expected_digests) or digest != expected_digests[index]
                    or not isinstance(reply, str) or len(reply) > 2500):
                raise ValueError('Checkpoint reply does not identify its submitted visual input.')
            replies.append(reply)
    return replies


def main(source, destination, folder, guide):
    from passenger_roadside35_native import generate, load_runtime

    model_folder = Path(folder)
    receipt = json.loads((model_folder / 'trial-download-verification.json').read_bytes())
    with tempfile.TemporaryDirectory() as local:
        copy_verified_weights(model_folder, Path(local), receipt['files'])
        print('Checkpoint files copied and verified against their original download hashes.', flush=True)
        processor, model = load_runtime(local)
        measure(json.loads(Path(source).read_bytes()), Path(guide).read_text(), Path(destination),
                lambda images, words: generate(processor, model, images, words))


if __name__ == '__main__':
    main(*sys.argv[1:])
