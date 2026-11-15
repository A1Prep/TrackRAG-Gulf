"""Prepare a bounded interval of original EMT images and human track labels."""

import argparse
import io
import json
import re
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from PIL import Image

REVISION = "4418cec01b03c6f797cadb06c9405843a30988e2"
BASE_URL = f"https://huggingface.co/datasets/KuAvLab/EMT/resolve/{REVISION}"


class _BoundedStream:
    def __init__(self, stream, limit):
        self.stream = stream
        self.remaining = limit

    def read(self, size=-1):
        size = self.remaining if size < 0 else min(size, self.remaining)
        payload = self.stream.read(size)
        self.remaining -= len(payload)
        return payload


def prepare_clip(data_root, recording_id, start_frame, count, limit_mib):
    if not re.fullmatch(r"video_[0-9]+(?:-[0-9]+)?", recording_id):
        raise ValueError("Use an EMT recording ID such as video_054604")
    if min(start_frame, count, limit_mib) < 1:
        raise ValueError("Frame numbers, frame count and download limit must be positive")
    destination = Path(data_root) / recording_id
    if destination.exists():
        raise FileExistsError(f"Already prepared: {destination}")
    wanted = {
        f"images/train/images/{recording_id}/{frame_id:05d}.jpg": frame_id
        for frame_id in range(start_frame, start_frame + count)
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        staging = Path(temporary) / "recording"
        (staging / "frames").mkdir(parents=True)
        frames = []
        request = urllib.request.Request(
            BASE_URL + "/train_images.tar.gz",
            headers={"Range": f"bytes=0-{limit_mib * 1024**2 - 1}"},
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            with tarfile.open(fileobj=_BoundedStream(response, limit_mib * 1024**2), mode="r|gz") as archive:
                for member in archive:
                    if member.name not in wanted or not member.isfile():
                        continue
                    frame_id = wanted.pop(member.name)
                    payload = archive.extractfile(member).read()
                    with Image.open(io.BytesIO(payload)) as image:
                        width, height = image.size
                        image.verify()
                    relative = f"frames/{frame_id:05d}.jpg"
                    (staging / relative).write_bytes(payload)
                    frames.append({
                        "recording_id": recording_id, "frame_id": frame_id,
                        "timestamp_s": (frame_id - 1) / 10,
                        "width": width, "height": height, "image_path": relative,
                    })
                    if len(frames) % 25 == 0:
                        print(f"Prepared {len(frames)} frames", flush=True)
                    if not wanted:
                        break
        if wanted:
            raise ValueError("Download limit reached before the requested training interval was found")
        frame_ids = {frame["frame_id"] for frame in frames}
        target = f"EMT/annotations/train/{recording_id}.txt"
        found = False
        observations = []
        with urllib.request.urlopen(BASE_URL + "/train_annotation.tar.gz", timeout=60) as response:
            with tarfile.open(fileobj=response, mode="r|gz") as archive:
                for member in archive:
                    if member.name != target:
                        continue
                    found = True
                    for line in archive.extractfile(member):
                        fields = line.decode("utf-8").split()
                        if not fields or int(fields[0]) not in frame_ids:
                            continue
                        observations.append({
                            "recording_id": recording_id, "frame_id": int(fields[0]),
                            "track_id": int(fields[1]), "class_name": fields[2],
                            "bbox_xyxy": list(map(float, fields[6:10])), "source": "annotated",
                        })
                    break
        if not found:
            raise ValueError("Recording annotations were not found in the pinned EMT training release")
        manifest = {
            "dataset": "KuAvLab/EMT", "revision": REVISION, "split": "train", "fps": 10,
            "time_origin_frame_id": 1, "recording_id": recording_id,
            "frames": sorted(frames, key=lambda frame: frame["frame_id"]),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        with (staging / "tracks.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for observation in observations:
                stream.write(json.dumps(observation) + "\n")
        staging.rename(destination)
    print(f"Saved {len(frames)} original frames and {len(observations)} boxes to {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--download-limit-mib", type=int, default=256)
    args = parser.parse_args()
    if min(args.start_frame, args.frames, args.download_limit_mib) < 1:
        parser.error("Frame numbers, frame count and download limit must be positive")
    prepare_clip(args.data_root, args.recording, args.start_frame, args.frames, args.download_limit_mib)


if __name__ == "__main__":
    main()
