"""Build the Task 1 fine-tuning dataset (COCO format) from the pinned EMT release archives.

Run on Colab after downloading the archives to local disk (see notebooks/task1_colab.ipynb):

    python -m tools.vision_prepare_dataset \
        --images-tar /content/emt/train_images.tar.gz \
        --annotations-tar /content/emt/train_annotation.tar.gz \
        --out /content/ds_v1 --pack /content/ds_v1.tar

Outputs (written to <out>.partial, renamed to <out> only after every check passes):
  train/_annotations.coco.json + JPEGs  Task 1 training recordings, every 5th frame
  valid/_annotations.coco.json + JPEGs  manifest dev recordings, every 5th frame
  dataset_manifest.json                 inputs, settings, counts, drops, statistics
  sanity/*.jpg                          random training images with their boxes drawn

The official test recordings are never read. Training excludes video_054604 (its last
frame is 3 s before dev video_054907 starts) and video_204347 (test video_204647 starts
when it ends): leakage control decided from recording timestamps, not from results.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import re
import shutil
import sys
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from trackrag.vision.classes import EMT_CLASSES, EMT_INDEX, NON_EMT_LABELS
from trackrag.vision.geometry import DEFAULT_LONG_SIDE, ResizeTransform, clip_box, resize_image

RELEASE = {
    "dataset": "KuAvLab/EMT",
    "revision": "4418cec01b03c6f797cadb06c9405843a30988e2",
    "train_images": "train_images.tar.gz",
    "train_images_bytes": 21848135924,
    "train_images_sha256": "d1b8d4221d9f2460d4243a5c1f2ed3133dc11231a256035c19c71a00165b3f1e",
    "train_annotations": "train_annotation.tar.gz",
}
EXCLUDED_FROM_TRAINING = {
    "video_054604": "ends 3 s before dev recording video_054907 starts",
    "video_204347": "official test recording video_204647 starts when it ends",
}
DEFAULT_MANIFEST = Path("splits") / "task2_manifest.json"
HEIGHT_BINS = (0, 4, 8, 16, 32, 64, 128, 256, 512)
OVERSHOOT_TOLERANCE_PX = 2.0  # real EMT labels overshoot the image by at most ~1 px

IMAGE_MEMBER = re.compile(r"^(?:\./)?images/(train|test)/images/([^/]+)/(\d+)\.(?:jpe?g|png)$", re.IGNORECASE)
LABEL_MEMBER = re.compile(r"^(?:\./)?(?:EMT/)?annotations/(train|test)/([^/]+?)(?:\.txt)?$")


def log(message: str) -> None:
    print(message, flush=True)


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, list[str]]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(manifest) != {"train", "dev", "test"}:
        raise ValueError("Split manifest must have exactly train, dev and test")
    owner: dict[str, str] = {}
    for split, ids in manifest.items():
        for recording in ids:
            if recording in owner:
                raise ValueError(f"{recording} appears in both {owner[recording]} and {split}")
            owner[recording] = split
    return manifest


def plan_roles(manifest: dict[str, list[str]], excluded: dict[str, str]) -> dict[str, str]:
    """Map recording -> 'train' or 'valid'. Test recordings get no role and are never read."""
    unknown = sorted(set(excluded) - set(manifest["train"]))
    if unknown:
        raise ValueError(f"Excluded recordings must be manifest training recordings: {unknown}")
    roles = {recording: "train" for recording in manifest["train"] if recording not in excluded}
    roles.update({recording: "valid" for recording in manifest["dev"]})
    leaked = sorted(set(roles) & set(manifest["test"]))
    if leaked:
        raise AssertionError(f"Test recordings would enter the fine-tuning data: {leaked}")
    if not any(role == "train" for role in roles.values()) or not any(role == "valid" for role in roles.values()):
        raise ValueError("Both training and validation recordings are required")
    return roles


def parse_label_rows(text: str, source: str):
    """Yield (frame_id, track_id, class_name, box) from an EMT tracking label file."""
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"[,\s]+", line)
        try:
            frame_value, track_value = float(parts[0]), float(parts[1])
            if not (frame_value.is_integer() and track_value.is_integer()) or min(frame_value, track_value) < 0:
                raise ValueError("frame and track ids must be nonnegative integers")
            box = tuple(float(value) for value in parts[6:10])
            if len(box) != 4 or not all(math.isfinite(value) for value in box):
                raise ValueError("box must have four finite values")
        except (IndexError, ValueError) as error:
            raise ValueError(f"{source}:{line_number}: invalid label row {raw!r}") from error
        yield int(frame_value), int(track_value), parts[2], box


def load_labels(annotations_tar: Path, roles: dict[str, str]):
    """Return {recording: {frame_id: [(track, class, box), ...]}} for recordings with a role."""
    labels: dict[str, dict[int, list]] = {}
    with tarfile.open(annotations_tar, mode="r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            match = LABEL_MEMBER.match(member.name)
            if not match or match.group(2) not in roles:
                continue
            split, recording = match.groups()
            if split != "train":
                raise AssertionError(f"Role recording {recording} found under annotations/{split}")
            if recording in labels:
                raise ValueError(f"Two label files for {recording}")
            text = archive.extractfile(member).read().decode("utf-8-sig")
            frames: dict[int, list] = defaultdict(list)
            seen = set()
            for frame_id, track_id, class_name, box in parse_label_rows(text, member.name):
                if (frame_id, track_id) in seen:
                    raise ValueError(f"{member.name}: duplicate track {track_id} in frame {frame_id}")
                seen.add((frame_id, track_id))
                frames[frame_id].append((track_id, class_name, box))
            labels[recording] = dict(frames)
    missing = sorted(set(roles) - set(labels))
    if missing:
        raise ValueError(f"No label files found for: {missing}")
    return labels


def height_bin(height: float) -> str:
    for low, high in zip(HEIGHT_BINS, HEIGHT_BINS[1:]):
        if low <= height < high:
            return f"{low}-{high}"
    return f">={HEIGHT_BINS[-1]}"


class DatasetWriter:
    def __init__(self, root: Path, long_side: int, jpeg_quality: int, min_box_px: float):
        self.root = root
        self.long_side = long_side
        self.jpeg_quality = jpeg_quality
        self.min_box_px = min_box_px
        self.coco = {role: {"images": [], "annotations": []} for role in ("train", "valid")}
        self.counts = {role: Counter() for role in ("train", "valid")}
        self.heights = {role: Counter() for role in ("train", "valid")}
        self.drops = Counter()
        for role in ("train", "valid"):
            (root / role).mkdir(parents=True)

    def add(self, role: str, recording: str, frame_id: int, payload: bytes, rows: list) -> tuple[int, int]:
        from PIL import Image

        with Image.open(io.BytesIO(payload)) as opened:
            opened.load()
            width, height = opened.size
            transform = ResizeTransform.for_size(width, height, self.long_side)
            small = resize_image(opened, transform)
        name = f"{recording}_{frame_id:05d}.jpg"
        destination = self.root / role / name
        temporary = destination.with_suffix(".tmp")
        small.save(temporary, format="JPEG", quality=self.jpeg_quality)
        temporary.replace(destination)
        images = self.coco[role]["images"]
        image_id = len(images) + 1
        images.append({
            "id": image_id, "file_name": name,
            "width": transform.resized_width, "height": transform.resized_height,
            "recording_id": recording, "frame_id": frame_id,
            "original_width": width, "original_height": height,
        })
        annotations = self.coco[role]["annotations"]
        for track_id, class_name, box in rows:
            if class_name not in EMT_INDEX:
                self.drops[f"non-EMT class {class_name}"] += 1
                continue
            left, top, right, bottom = box
            tolerance = OVERSHOOT_TOLERANCE_PX
            if left < -tolerance or top < -tolerance or right > width + tolerance or bottom > height + tolerance:
                raise ValueError(
                    f"{recording} frame {frame_id}: box {box} lies more than {tolerance:g} px outside the "
                    f"{width}x{height} image - labels and images do not use the same coordinates"
                )
            clipped = clip_box(box, width, height)
            if clipped is None:
                self.drops["no area inside the image"] += 1
                continue
            if clipped != box:
                self.drops["clipped to the image (kept)"] += 1
            left, top, right, bottom = transform.to_resized(clipped)
            box_w, box_h = right - left, bottom - top
            if box_w < self.min_box_px or box_h < self.min_box_px:
                self.drops[f"smaller than {self.min_box_px:g} px after resizing"] += 1
                continue
            annotations.append({
                "id": len(annotations) + 1, "image_id": image_id,
                "category_id": EMT_INDEX[class_name] + 1,
                "bbox": [left, top, box_w, box_h], "area": box_w * box_h, "iscrowd": 0,
                "track_id": track_id,
            })
            self.counts[role][class_name] += 1
            self.heights[role][height_bin(box_h)] += 1
        return width, height

    def write_annotations(self) -> None:
        categories = [{"id": index + 1, "name": name, "supercategory": "none"} for index, name in enumerate(EMT_CLASSES)]
        for role, data in self.coco.items():
            document = {
                "info": {"description": f"EMT Task 1 fine-tuning {role} split", "release": RELEASE["revision"]},
                "licenses": [{"id": 1, "name": "CC BY-NC-SA 4.0 (EMT dataset card)"}],
                "categories": categories, **data,
            }
            path = self.root / role / "_annotations.coco.json"
            path.write_text(json.dumps(document, allow_nan=False), encoding="utf-8")


def draw_sanity(root: Path, writer: DatasetWriter, count: int, seed: int) -> list[str]:
    from PIL import Image, ImageDraw

    images = writer.coco["train"]["images"]
    if not images or count < 1:
        return []
    by_image = defaultdict(list)
    for annotation in writer.coco["train"]["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    target = root / "sanity"
    target.mkdir()
    written = []
    for image in random.Random(seed).sample(images, min(count, len(images))):
        with Image.open(root / "train" / image["file_name"]) as opened:
            canvas = opened.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        for annotation in by_image[image["id"]]:
            x, y, w, h = annotation["bbox"]
            draw.rectangle((x, y, x + w, y + h), outline=(255, 0, 0), width=2)
            draw.text((x, max(0, y - 10)), EMT_CLASSES[annotation["category_id"] - 1], fill=(255, 255, 0))
        canvas.save(target / image["file_name"], quality=90)
        written.append(image["file_name"])
    return written


def prepare(images_tar, annotations_tar, out, *, manifest_path=DEFAULT_MANIFEST, excluded=None,
            train_every=5, valid_every=5, long_side=DEFAULT_LONG_SIDE, jpeg_quality=95, min_box_px=1.0,
            images_sha256=RELEASE["train_images_sha256"], sanity=10, seed=0, pack=None) -> dict:
    images_tar, annotations_tar, out = Path(images_tar), Path(annotations_tar), Path(out)
    excluded = dict(EXCLUDED_FROM_TRAINING if excluded is None else excluded)
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}")
    if pack is not None:
        pack = Path(pack)
        for target in (pack, pack.with_name(pack.name + ".sha256")):
            if target.exists():
                raise FileExistsError(f"Pack target already exists: {target}")
    if min(train_every, valid_every, long_side) < 1 or not 1 <= jpeg_quality <= 100:
        raise ValueError("Invalid sampling, size or quality setting")
    started = time.time()
    manifest = load_manifest(manifest_path)
    roles = plan_roles(manifest, excluded)
    log(f"Training recordings ({sum(r == 'train' for r in roles.values())}): {sorted(r for r, v in roles.items() if v == 'train')}")
    log(f"Validation recordings: {sorted(r for r, v in roles.items() if v == 'valid')}")
    log(f"Excluded from training: {excluded}")

    inputs = {"images_tar": {"path": str(images_tar), "bytes": images_tar.stat().st_size},
              "annotations_tar": {"path": str(annotations_tar), "bytes": annotations_tar.stat().st_size,
                                  "sha256": sha256_file(annotations_tar)}}
    if images_sha256:
        log("Verifying image archive sha256 (reads the whole file once)...")
        actual = sha256_file(images_tar)
        if actual != images_sha256:
            raise ValueError(f"Image archive sha256 {actual} != expected {images_sha256}")
        inputs["images_tar"]["sha256"] = actual
        log("Image archive sha256 verified.")
    else:
        inputs["images_tar"]["sha256"] = "not verified"

    labels = load_labels(annotations_tar, roles)
    log(f"Loaded labels for {len(labels)} recordings.")

    partial = out.with_name(out.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    writer = DatasetWriter(partial, long_side, jpeg_quality, min_box_px)
    every = {"train": train_every, "valid": valid_every}
    inventory: dict[str, list[int]] = defaultdict(list)
    sizes: dict[str, Counter] = defaultdict(Counter)
    other_members = Counter()
    skipped_unlabelled: Counter = Counter()
    processed = 0
    with tarfile.open(images_tar, mode="r|gz", bufsize=1 << 20) as archive:
        for member in archive:
            if not member.isfile():
                continue
            match = IMAGE_MEMBER.match(member.name)
            if not match:
                other_members["unrecognised member"] += 1
                continue
            split, recording, frame_text = match.groups()
            role = roles.get(recording)
            if role is None:
                other_members[f"{split} recording without a role"] += 1
                continue
            if split != "train":
                raise AssertionError(f"Role recording {recording} found under images/{split}")
            frame_id = int(frame_text)
            inventory[recording].append(frame_id)
            if (frame_id - 1) % every[role]:
                continue
            rows = labels[recording].get(frame_id)
            if not rows:
                # Never train on a frame whose labels are missing: it would teach "no objects here".
                skipped_unlabelled[recording] += 1
                continue
            payload = archive.extractfile(member).read()
            size = writer.add(role, recording, frame_id, payload, rows)
            sizes[recording][f"{size[0]}x{size[1]}"] += 1
            processed += 1
            if processed % 500 == 0:
                log(f"Prepared {processed} images ({time.time() - started:.0f}s)")

    problems = []
    for recording in sorted(roles):
        frames = inventory.get(recording, [])
        if not frames:
            problems.append(f"{recording}: no images in the archive")
            continue
        if len(frames) != len(set(frames)):
            problems.append(f"{recording}: duplicate frame images")
        if len(sizes[recording]) != 1:
            problems.append(f"{recording}: mixed image sizes {dict(sizes[recording])}")
        # A truncated or mismatched archive shows up as labelled frames without an image.
        missing_images = sorted(set(labels[recording]) - set(frames))
        if missing_images:
            problems.append(f"{recording}: {len(missing_images)} labelled frames have no image "
                            f"(first {missing_images[:5]}); archive truncated or mismatched")
    if problems:
        shutil.rmtree(partial)
        raise ValueError("Dataset checks failed: " + "; ".join(problems))
    for role in ("train", "valid"):
        if not writer.coco[role]["images"]:
            shutil.rmtree(partial)
            raise ValueError(f"No {role} images were written")
    writer.write_annotations()
    sanity_files = draw_sanity(partial, writer, sanity, seed)

    recordings = {}
    for recording in sorted(roles):
        frames = sorted(inventory[recording])
        gaps = [(a, b) for a, b in zip(frames, frames[1:]) if b != a + 1]
        all_labels = Counter(row[1] for rows in labels[recording].values() for row in rows)
        recordings[recording] = {
            "role": roles[recording], "frames_in_archive": len(frames), "first_frame": frames[0],
            "last_frame": frames[-1], "frame_gaps": gaps[:20], "frame_gap_count": len(gaps),
            "image_size": next(iter(sizes[recording])), "all_label_rows_by_class": dict(sorted(all_labels.items())),
            "images_without_labels": len(set(frames) - set(labels[recording])),
            "sampled_images_skipped_without_labels": skipped_unlabelled[recording],
        }
    summary = {
        "version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "release": RELEASE, "inputs": inputs, "split_manifest": str(manifest_path),
        "excluded_from_training": excluded,
        "settings": {"train_every": train_every, "valid_every": valid_every, "frame_rule": "(frame_id - 1) % every == 0",
                     "long_side": long_side, "resample": "PIL BILINEAR (antialiased downscale), downscale only",
                     "jpeg_quality": jpeg_quality, "min_box_px_after_resize": min_box_px,
                     "categories": list(EMT_CLASSES), "non_emt_labels_expected": list(NON_EMT_LABELS)},
        "counts": {role: {"images": len(writer.coco[role]["images"]),
                          "annotations": len(writer.coco[role]["annotations"]),
                          "by_class": {name: writer.counts[role][name] for name in EMT_CLASSES},
                          "box_height_px_after_resize": dict(sorted(writer.heights[role].items()))}
                   for role in ("train", "valid")},
        "dropped_or_adjusted_label_rows": dict(sorted(writer.drops.items())),
        "sampled_images_skipped_without_labels": sum(skipped_unlabelled.values()),
        "other_archive_members": dict(other_members),
        "recordings": recordings, "sanity_images": sanity_files,
        "seconds": round(time.time() - started, 1),
    }
    (partial / "dataset_manifest.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    partial.rename(out)
    log(f"Dataset written to {out}: {summary['counts']['train']['images']} train / "
        f"{summary['counts']['valid']['images']} valid images.")
    if pack:
        pack = Path(pack)
        if pack.exists():
            raise FileExistsError(f"Pack target already exists: {pack}")
        temporary = pack.with_name(pack.name + ".partial")
        with tarfile.open(temporary, mode="w") as archive:
            archive.add(out, arcname=out.name)
        temporary.replace(pack)
        digest = sha256_file(pack)
        pack.with_name(pack.name + ".sha256").write_text(f"{digest}  {pack.name}\n", encoding="utf-8")
        summary["pack"] = {"path": str(pack), "sha256": digest, "bytes": pack.stat().st_size}
        log(f"Packed {pack} sha256={digest}")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-tar", type=Path, required=True)
    parser.add_argument("--annotations-tar", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--train-every", type=int, default=5)
    parser.add_argument("--valid-every", type=int, default=5)
    parser.add_argument("--long-side", type=int, default=DEFAULT_LONG_SIDE)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--min-box-px", type=float, default=1.0)
    parser.add_argument("--skip-sha256", action="store_true", help="Skip verifying the image archive hash")
    parser.add_argument("--sanity", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pack", type=Path, help="Also write an uncompressed tar of the dataset (+ .sha256)")
    args = parser.parse_args(argv)
    prepare(args.images_tar, args.annotations_tar, args.out, manifest_path=args.manifest,
            train_every=args.train_every, valid_every=args.valid_every, long_side=args.long_side,
            jpeg_quality=args.jpeg_quality, min_box_px=args.min_box_px,
            images_sha256=None if args.skip_sha256 else RELEASE["train_images_sha256"],
            sanity=args.sanity, seed=args.seed, pack=args.pack)
    return 0


if __name__ == "__main__":
    sys.exit(main())
