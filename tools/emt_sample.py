"""Build the shared three-frame example from a pinned EMT training release."""
import io
import json
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REVISION = "4418cec01b03c6f797cadb06c9405843a30988e2"
BASE_URL = f"https://huggingface.co/datasets/KuAvLab/EMT/resolve/{REVISION}"
RECORDING = "video_054604"
FRAME_IDS = (1, 2, 3)
FPS = 10
DESTINATION = Path(__file__).resolve().parents[1] / "data" / "emt-sample"


def archive_prefix(name, mib):
    limit = mib * 1024 ** 2
    request = urllib.request.Request(
        f"{BASE_URL}/{name}", headers={"Range": f"bytes=0-{limit - 1}"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read(limit)
    return tarfile.open(fileobj=io.BytesIO(payload), mode="r|gz")


def extract_images(directory):
    wanted = {
        f"images/train/images/{RECORDING}/{frame_id:05d}.jpg": frame_id
        for frame_id in FRAME_IDS
    }
    with archive_prefix("train_images.tar.gz", 4) as archive:
        for member in archive:
            if member.name not in wanted or not member.isfile():
                continue
            frame_id = wanted.pop(member.name)
            image = archive.extractfile(member).read()
            if not image.startswith(b"\xff\xd8") or not image.endswith(b"\xff\xd9"):
                raise ValueError(f"Invalid JPEG for frame {frame_id}")
            (directory / f"{frame_id:05d}.jpg").write_bytes(image)
            if not wanted:
                return
    raise ValueError(f"Missing frames in the pinned archive: {sorted(wanted)}")


def read_tracks():
    tracks = {frame_id: {} for frame_id in FRAME_IDS}
    target = f"EMT/annotations/train/{RECORDING}.txt"
    with archive_prefix("train_annotation.tar.gz", 8) as archive:
        for member in archive:
            if member.name != target or not member.isfile():
                continue
            for line in archive.extractfile(member):
                columns = line.decode("utf-8").split()
                if not columns or int(columns[0]) not in tracks:
                    continue
                frame_id, track_id = map(int, columns[:2])
                x1, y1, x2, y2 = map(float, columns[6:10])
                if x1 >= x2 or y1 >= y2 or track_id in tracks[frame_id]:
                    raise ValueError(f"Invalid box or duplicate track at frame {frame_id}")
                tracks[frame_id][track_id] = {
                    "track_id": track_id, "class_name": columns[2],
                    "bbox_xyxy": [x1, y1, x2, y2],
                }
            break
    if not all(tracks.values()):
        raise ValueError("Missing EMT tracks for one or more sample frames")
    return tracks


def main():
    if DESTINATION.exists():
        print(f"Using the existing sample at {DESTINATION}")
        return
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=DESTINATION.parent) as temp_dir:
        sample = Path(temp_dir) / "sample"
        sample.mkdir()
        extract_images(sample)
        tracks = read_tracks()
        frames = [{
            "frame_id": frame_id, "timestamp_s": (frame_id - FRAME_IDS[0]) / FPS,
            "image": f"{frame_id:05d}.jpg",
            "objects": [tracks[frame_id][key] for key in sorted(tracks[frame_id])],
        } for frame_id in FRAME_IDS]
        manifest = {
            "dataset_revision": REVISION, "split": "train", "recording_id": RECORDING,
            "source": "emt_ground_truth", "fps": FPS,
            "time_origin_frame_id": FRAME_IDS[0], "frames": frames,
        }
        (sample / "tracks.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        sample.rename(DESTINATION)
    boxes = sum(len(objects) for objects in tracks.values())
    print(f"{RECORDING}: {len(frames)} frames, {boxes} labeled boxes")
    print(DESTINATION)


if __name__ == "__main__":
    main()
