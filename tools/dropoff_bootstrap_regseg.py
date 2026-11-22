"""Bounded, training-only Talk2Car-RegSeg pilot download.

This is a small real-data bootstrap, not the final grouped corpus.  It never
downloads the complete 1.49 GB release.  The outer ZIP contains deflated TARs,
so the image reader deliberately consumes a bounded prefix rather than
pretending individual TAR images can be fetched independently.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import struct
import tarfile
from urllib.request import Request, urlopen
import zlib


SOURCE_REVISION = "cebce1943ce373857cc9be5def68ded4008c29e1"
RAW_ROOT = f"https://raw.githubusercontent.com/kanji1011/Talk2car-Refseg/{SOURCE_REVISION}"
METADATA_URL = f"{RAW_ROOT}/dataloader/talk2car_w_rpn_no_duplicates.json"
METADATA_SIZE = 78_543_821
METADATA_SHA256 = "c94b741178b0cf8955014e43229fd0eb9afb384c29269b7c071e9858b402a3f8"
ARCHIVE_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1kNEtbmV_777l8FxYig9WnxYH_E6W-4jH&export=download&confirm=t"
)
ARCHIVE_SIZE = 1_485_242_059
ARCHIVE_DIRECTORY_SHA256 = "32fe81c35bed2cac8b675fdeed7421bd39a2c8f40dd8e148c6cd4a2567283134"
MASK_MEMBER_COMPRESSED_SHA256 = "cd1b6320199a09020b1a10047e3d6e44e0b27328ec6d8f54063c3ac9ac2b1092"
TRAIN_REWRITE_URL = f"{RAW_ROOT}/dataloader/annotation_new_train.txt"
TRAIN_REWRITE_SHA256 = "45fa1baa6f1a48d2b300ac3604016e4cc1c8bc29ec7b8723068998ea3fd10653"
VAL_REWRITE_URL = f"{RAW_ROOT}/dataloader/annotation_new_val.txt"
VAL_REWRITE_SHA256 = "052c34427f228f06094605f2133a28073a7908b4445a49747fef7ae29a5d7a15"
TALK2CAR_REVISION = "3fde52aee3341312196ab9b2f999337eff0693a0"
SCENE_METADATA_URL = (
    f"https://raw.githubusercontent.com/talk2car/Talk2Car/{TALK2CAR_REVISION}"
    "/data/commands/train_commands.json"
)
SCENE_METADATA_SIZE = 4_891_514
SCENE_METADATA_SHA256 = "5a90c471284685bd8dd8bfa6e992cfefce69e58a392c934d2f9d54ea119a42d2"


class BootstrapError(ValueError):
    """A source, download budget or annotation validation failed."""


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


@dataclass
class BoundedSource:
    """Only accept exact HTTP byte ranges, with a cumulative byte ceiling."""

    max_bytes: int
    opener: object = urlopen
    used_bytes: int = 0
    receipts: list[dict] = field(default_factory=list)

    def fetch(self, url: str, start: int, end: int, *, total: int | None = None) -> bytes:
        if start < 0 or end < start:
            raise BootstrapError("Invalid byte range")
        length = end - start + 1
        if self.used_bytes + length > self.max_bytes:
            raise BootstrapError("Cumulative source download budget exceeded")
        request = Request(url, headers={
            "Range": f"bytes={start}-{end}", "Accept-Encoding": "identity",
        })
        with self.opener(request, timeout=90) as response:
            if response.status != 206:
                raise BootstrapError("Server ignored the bounded range request")
            content_range = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
            if not match or (int(match[1]), int(match[2])) != (start, end):
                raise BootstrapError("Unexpected Content-Range")
            if total is not None and int(match[3]) != total:
                raise BootstrapError("Source total length changed")
            if response.headers.get("Content-Encoding", "identity") != "identity":
                raise BootstrapError("Unexpected transport content encoding")
            blob = response.read(length + 1)
        self.used_bytes += len(blob)
        if len(blob) != length:
            raise BootstrapError("Truncated or oversized range response")
        self.receipts.append({
            "url": url, "start": start, "end": end,
            "source_size": int(match[3]), "bytes": len(blob), "sha256": sha256(blob),
        })
        return blob

    def small_file(self, url: str, size: int, expected_hash: str) -> bytes:
        blob = self.fetch(url, 0, size - 1, total=size)
        if sha256(blob) != expected_hash:
            raise BootstrapError("Pinned source SHA256 mismatch")
        return blob


def excluded_source_ids(blob: bytes) -> set[int]:
    result = set()
    for line in blob.decode("utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not parts[0].isdigit():
            raise BootstrapError("Malformed concise-test source mapping")
        value = int(parts[0])
        if value in result:
            raise BootstrapError("Duplicate concise-test source mapping")
        result.add(value)
    return result


def verified_scene_map(metadata: dict, source: dict, concise_train_ids: set[int]) -> tuple[dict[int, dict], set[str]]:
    """Join two pinned releases and exclude all scenes reused by concise test."""
    train = metadata.get("train")
    commands = source.get("commands")
    if not isinstance(train, dict) or not isinstance(commands, list):
        raise BootstrapError("Invalid source metadata structure")
    by_token = {}
    for row in commands:
        if not isinstance(row, dict) or not isinstance(row.get("command_token"), str):
            raise BootstrapError("Missing source command token")
        token = row["command_token"]
        if token in by_token:
            raise BootstrapError("Duplicate source command token")
        by_token[token] = row
    result = {}
    for identity, row in train.items():
        if not isinstance(identity, str) or not identity.isdigit() or not isinstance(row, dict):
            raise BootstrapError("Invalid RegSeg training record")
        match = by_token.get(row.get("command_token"))
        if match is None or match.get("t2c_img") != row.get("img") or match.get("command") != row.get("command"):
            raise BootstrapError("RegSeg/Talk2Car command-image join mismatch")
        if not all(isinstance(match.get(key), str) and re.fullmatch(r"[0-9a-f]{32}", match[key])
                   for key in ("scene_token", "sample_token")):
            raise BootstrapError("Invalid source scene/sample token")
        result[int(identity)] = {
            "scene_token": match["scene_token"], "sample_token": match["sample_token"],
        }
    if not concise_train_ids.issubset(result):
        raise BootstrapError("Concise-test source IDs absent from verified training map")
    excluded_scenes = {result[identity]["scene_token"] for identity in concise_train_ids}
    return result, excluded_scenes


def zip_directory(tail: bytes, total: int) -> dict[str, dict]:
    """Parse this small classic ZIP central directory, refusing unsafe variants."""
    position = tail.rfind(b"PK\x05\x06")
    if position < 0 or len(tail) - position < 22:
        raise BootstrapError("ZIP end record missing")
    record = struct.unpack("<4s4H2IH", tail[position:position + 22])
    _, disk, cd_disk, disk_count, count, cd_size, cd_offset, comment_size = record
    if disk or cd_disk or disk_count != count or count == 65535:
        raise BootstrapError("Unsupported split/ZIP64 archive")
    if position + 22 + comment_size != len(tail):
        raise BootstrapError("Unexpected ZIP trailing data")
    start = cd_offset - (total - len(tail))
    if start < 0 or start + cd_size != position:
        raise BootstrapError("Central directory is outside bounded tail")
    cursor = start
    members = {}
    for _ in range(count):
        if tail[cursor:cursor + 4] != b"PK\x01\x02":
            raise BootstrapError("Invalid ZIP directory entry")
        header = struct.unpack("<4s6H3I5H2I", tail[cursor:cursor + 46])
        _, _, _, flags, method, _, _, crc, compressed, size, nlen, xlen, clen, _, _, _, offset = header
        name = tail[cursor + 46:cursor + 46 + nlen].decode("utf-8")
        safe_name(name)
        if flags & 1 or method not in (0, 8) or name in members:
            raise BootstrapError("Encrypted, duplicate or unsupported ZIP member")
        members[name] = {
            "offset": offset, "compressed_size": compressed, "size": size,
            "crc32": crc, "compression": method,
        }
        cursor += 46 + nlen + xlen + clen
    if cursor != start + cd_size:
        raise BootstrapError("ZIP directory length mismatch")
    return members


def safe_name(name: str) -> None:
    path = PurePosixPath(name)
    if not name or "\\" in name or path.is_absolute() or ".." in path.parts or ":" in name:
        raise BootstrapError("Unsafe archive member name")


def read_outer_member(
    source: BoundedSource, name: str, entry: dict, *,
    prefix_bytes: int | None = None, max_uncompressed: int = 64 * 1024 * 1024,
) -> tuple[bytes, dict]:
    start = entry["offset"]
    header = source.fetch(ARCHIVE_URL, start, start + 29, total=ARCHIVE_SIZE)
    values = struct.unpack("<4s5H3I2H", header)
    signature, _, flags, method, _, _, crc, compressed, size, nlen, xlen = values
    if signature != b"PK\x03\x04" or flags & 1:
        raise BootstrapError("Invalid ZIP local header")
    names = source.fetch(ARCHIVE_URL, start + 30, start + 29 + nlen + xlen, total=ARCHIVE_SIZE)
    if names[:nlen].decode("utf-8") != name:
        raise BootstrapError("ZIP local member identity mismatch")
    if (method, compressed, size, crc) != (
        entry["compression"], entry["compressed_size"], entry["size"], entry["crc32"],
    ):
        raise BootstrapError("ZIP local/directory metadata mismatch")
    count = compressed if prefix_bytes is None else min(prefix_bytes, compressed)
    data_start = start + 30 + nlen + xlen
    blob = source.fetch(ARCHIVE_URL, data_start, data_start + count - 1, total=ARCHIVE_SIZE)
    if method == 8:
        decoder = zlib.decompressobj(-15)
        content = decoder.decompress(blob, max_uncompressed + 1)
        if len(content) > max_uncompressed or decoder.unconsumed_tail:
            raise BootstrapError("Uncompressed ZIP member exceeds safety limit")
        complete = decoder.eof
    else:
        content, complete = blob, count == compressed
    if count == compressed:
        if not complete or len(content) != size or zlib.crc32(content) & 0xFFFFFFFF != crc:
            raise BootstrapError("Complete ZIP member size/CRC mismatch")
    return content, {
        "member": name, "compressed_bytes_read": count, "complete": complete,
        "compressed_sha256": sha256(blob), "uncompressed_prefix_sha256": sha256(content),
        "uncompressed_bytes_available": len(content),
    }


def tar_files(blob: bytes, pattern: str, *, partial: bool = False) -> dict[int, bytes]:
    """Read exact selected regular files, never extract paths from an archive."""
    result = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r|") as archive:
            for member in archive:
                safe_name(member.name)
                match = re.fullmatch(pattern, member.name)
                if not match:
                    continue
                if not member.isfile() or member.size > 10 * 1024 * 1024:
                    raise BootstrapError("Invalid or oversized selected TAR member")
                identity = int(match[1])
                if identity in result:
                    raise BootstrapError("Duplicate selected TAR member")
                stream = archive.extractfile(member)
                if stream is None:
                    raise BootstrapError("Selected TAR member has no content")
                content = stream.read(member.size + 1)
                if len(content) != member.size:
                    raise BootstrapError("Truncated selected TAR member")
                result[identity] = content
    except tarfile.ReadError:
        if not partial:
            raise BootstrapError("Truncated TAR member") from None
    return result


def mask_targets(image_blob: bytes, mask_blob: bytes) -> tuple[tuple[int, int], list[float], list[float], tuple[int, int]]:
    """Map native mask pixels to the untouched image, without mask resampling."""
    from PIL import Image

    with Image.open(io.BytesIO(image_blob)) as image:
        image.load()
        width, height = image.size
        if image.format != "JPEG" or width <= 0 or height <= 0:
            raise BootstrapError("Invalid original JPEG")
    with Image.open(io.BytesIO(mask_blob)) as original_mask:
        if original_mask.format != "PNG":
            raise BootstrapError("Invalid PNG mask")
        mask = original_mask.convert("L")
        mask.load()
    # The official loader uses mask > 0; do not use interpolated mask values.
    foreground = mask.point(lambda value: 255 if value > 0 else 0)
    bounds = foreground.getbbox()
    if bounds is None:
        raise BootstrapError("Empty destination mask")
    left, top, right, bottom = bounds
    mw, mh = mask.size
    cx, cy = (left + right - 1) / 2, (top + bottom - 1) / 2
    pixels = foreground.load()
    best = None
    for y in range(top, bottom):
        for x in range(left, right):
            if pixels[x, y]:
                candidate = ((x - cx) ** 2 + (y - cy) ** 2, y, x)
                if best is None or candidate < best:
                    best = candidate
    _, y, x = best
    point = [(x + 0.5) * width / mw, (y + 0.5) * height / mh]
    box = [left * width / mw, top * height / mh, right * width / mw, bottom * height / mh]
    return (width, height), point, box, (mw, mh)


def build_cases(
    metadata: dict, images: dict[int, bytes], masks: dict[int, bytes],
    excluded: set[int], output: Path, *, limit: int, source_receipt: dict,
    scene_map: dict[int, dict] | None = None, excluded_scenes: set[str] | None = None,
    rejected_records: list[dict] | None = None,
) -> list[dict]:
    if output.exists():
        raise BootstrapError("Refusing to overwrite an existing bootstrap directory")
    if limit < 1:
        raise BootstrapError("limit must be positive")
    selected = [identity for identity in images if identity in masks and identity not in excluded]
    if scene_map is not None:
        if any(identity not in scene_map for identity in selected):
            raise BootstrapError("Image absent from verified source scene mapping")
        selected = [identity for identity in selected if scene_map[identity]["scene_token"] not in (excluded_scenes or set())]
    if not selected:
        raise BootstrapError("No safe joined training records in image prefix")
    cases = []
    prepared = []
    for identity in selected:
        if len(cases) >= limit:
            break
        row = metadata.get("train", {}).get(str(identity))
        if not isinstance(row, dict) or row.get("img") != f"img_train_{identity}.jpg":
            raise BootstrapError("Training metadata/image identity mismatch")
        command = row.get("command")
        if not isinstance(command, str) or not command.strip():
            raise BootstrapError("Missing source command")
        try:
            shape, point, box, mask_shape = mask_targets(images[identity], masks[identity])
        except (BootstrapError, OSError) as error:
            if rejected_records is None:
                raise
            rejected_records.append({
                "source_index": identity,
                "source_image": f"img_train_{identity}.jpg",
                "reason": str(error), "error_type": type(error).__name__,
                "image_sha256": sha256(images[identity]),
                "mask_sha256": sha256(masks[identity]),
            })
            continue
        case_id = f"regseg-train-{identity}"
        image_relative = Path("images") / f"img_train_{identity}.jpg"
        mask_relative = Path("masks") / f"gt_img_ann_train_{identity}.png"
        image_path = output / image_relative
        mask_path = output / mask_relative
        source = {
            **source_receipt, "dataset": "Talk2Car-RegSeg", "source_split": "train",
            "source_index": identity, "command_token": row.get("command_token"),
            "grouping_level": "renamed_source_image_only",
            "scene_group_verified": False,
            "usage": "training_pilot_only_not_scene_independent_evaluation",
            "image_sha256": sha256(images[identity]), "mask_sha256": sha256(masks[identity]),
            "mask_path": mask_relative.as_posix(), "mask_width": mask_shape[0], "mask_height": mask_shape[1],
            "target_rule": "mask>0; pixel-centre nearest foreground-bbox centre; scaled native mask bbox",
        }
        scene_group = f"regseg-image-only:img_train_{identity}"
        if scene_map is not None:
            source.update({
                **scene_map[identity], "scene_metadata_revision": TALK2CAR_REVISION,
                "scene_metadata_sha256": SCENE_METADATA_SHA256,
                "grouping_level": "nuscenes_source_scene", "scene_group_verified": True,
                "concise_test_source_scene_excluded": False,
            })
            scene_group = f"talk2car-scene:{scene_map[identity]['scene_token']}"
        cases.append({
            "schema_version": 1, "case_id": case_id,
            "scene_group": scene_group,
            "partition": "train", "source": source,
            "request": {
                "request_id": case_id,
                "scene": {
                    "frame": {
                        "recording_id": f"regseg-train-{identity}", "frame_id": 0,
                        "timestamp_s": 0.0, "width": shape[0], "height": shape[1],
                        "image_path": image_relative.as_posix(),
                    },
                    "source": "predicted", "tracks": [],
                    "provenance": {"mode": "image_only_public_training", "detector_run": False},
                },
                "text": command, "input_mode": "typed", "dialogue": [],
            },
            "target": {"destination_point_xy": point, "destination_box_xyxy": box},
            "supervised_fields": ["destination_point_xy", "destination_box_xyxy"],
            "review": {"status": "public_annotation", "new_human_review": False},
        })
        prepared.append((image_path, images[identity], mask_path, masks[identity]))
    if not cases:
        raise BootstrapError("No valid destination records in the safe image prefix")
    output.mkdir(parents=True)
    (output / "images").mkdir()
    (output / "masks").mkdir()
    for image_path, image_blob, mask_path, mask_blob in prepared:
        image_path.write_bytes(image_blob)
        mask_path.write_bytes(mask_blob)
    (output / "cases.jsonl").write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases), encoding="utf-8",
    )
    return cases


def bootstrap(output: Path, *, limit: int = 8, max_download_mib: int = 128, image_prefix_mib: int = 4) -> dict:
    if output.exists():
        raise BootstrapError("Output already exists")
    if not 1 <= image_prefix_mib <= 32 or not 1 <= limit <= 100:
        raise BootstrapError("Pilot image prefix/record limit is out of bounds")
    if not METADATA_SHA256:
        raise BootstrapError("Pinned metadata hash has not been configured")
    source = BoundedSource(max_download_mib * 1024 * 1024)
    metadata_blob = source.small_file(METADATA_URL, METADATA_SIZE, METADATA_SHA256)
    metadata = json.loads(metadata_blob)
    train_rewrites = source.small_file(TRAIN_REWRITE_URL, 7211, TRAIN_REWRITE_SHA256)
    val_rewrites = source.small_file(VAL_REWRITE_URL, 13872, VAL_REWRITE_SHA256)
    excluded = excluded_source_ids(train_rewrites)
    excluded_val = excluded_source_ids(val_rewrites)
    scene_metadata = json.loads(source.small_file(
        SCENE_METADATA_URL, SCENE_METADATA_SIZE, SCENE_METADATA_SHA256,
    ))
    scenes, excluded_scenes = verified_scene_map(metadata, scene_metadata, excluded)
    tail = source.fetch(ARCHIVE_URL, ARCHIVE_SIZE - 65557, ARCHIVE_SIZE - 1, total=ARCHIVE_SIZE)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 313 or sha256(tail[eocd - 313:eocd]) != ARCHIVE_DIRECTORY_SHA256:
        raise BootstrapError("Pinned archive central directory changed")
    directory = zip_directory(tail, ARCHIVE_SIZE)
    required = ("Talk2Car-RefSeg/images.tar", "Talk2Car-RefSeg/masks.tar")
    if any(name not in directory for name in required):
        raise BootstrapError("Expected release members missing")
    image_tar, image_receipt = read_outer_member(
        source, required[0], directory[required[0]], prefix_bytes=image_prefix_mib * 1024 * 1024,
    )
    mask_tar, mask_receipt = read_outer_member(source, required[1], directory[required[1]])
    if mask_receipt["compressed_sha256"] != MASK_MEMBER_COMPRESSED_SHA256:
        raise BootstrapError("Pinned mask member SHA256 mismatch")
    images = tar_files(image_tar, r"imgs/img_train_(\d+)\.jpg", partial=True)
    masks = tar_files(mask_tar, r"(?:masks|val_masks_new)/gt_img_ann_train_(\d+)\.png")
    provenance = {
        "repository_revision": SOURCE_REVISION, "metadata_url": METADATA_URL,
        "metadata_sha256": METADATA_SHA256, "archive_url": ARCHIVE_URL,
        "archive_size": ARCHIVE_SIZE, "archive_full_sha256_verified": False,
        "concise_test_train_exclusions_sha256": TRAIN_REWRITE_SHA256,
        "scene_metadata_url": SCENE_METADATA_URL,
        "scene_metadata_sha256": SCENE_METADATA_SHA256,
        "excluded_concise_test_train_scene_count": len(excluded_scenes),
    }
    rejected_records = []
    cases = build_cases(
        metadata, images, masks, excluded, output, limit=limit,
        source_receipt=provenance, scene_map=scenes, excluded_scenes=excluded_scenes,
        rejected_records=rejected_records,
    )
    receipt = {
        "purpose": "real_training_only_bootstrap_not_final_evaluation",
        "case_count": len(cases), "requested_case_count": limit,
        "rejected_source_record_count": len(rejected_records),
        "rejected_source_records": rejected_records,
        "source_train_count": len(metadata.get("train", {})),
        "excluded_concise_test_source_train_ids": sorted(excluded),
        "excluded_concise_test_source_val_ids": sorted(excluded_val),
        "excluded_concise_test_source_train_scenes": sorted(excluded_scenes),
        "excluded_train_ids_by_scene": sorted(identity for identity, row in scenes.items() if row["scene_token"] in excluded_scenes),
        "image_member": image_receipt, "mask_member": mask_receipt,
        "downloaded_bytes": source.used_bytes, "download_budget_bytes": source.max_bytes,
        "downloads": source.receipts, "provenance": provenance,
        "scene_group_mapping_verified": True,
        "warning": "Official create_dataset.py rewrites both train and val source IDs into concise test. Every training scene represented by those train IDs is excluded. This pilot contains training cases only, not scene-independent evaluation.",
    }
    (output / "bootstrap-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--max-download-mib", type=int, default=128)
    parser.add_argument("--image-prefix-mib", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        receipt = bootstrap(
            args.out, limit=args.limit, max_download_mib=args.max_download_mib,
            image_prefix_mib=args.image_prefix_mib,
        )
    except (BootstrapError, OSError, ValueError, zlib.error) as error:
        parser.exit(2, f"RegSeg bootstrap failed: {error}\n")
    print(json.dumps({"cases": receipt["case_count"], "downloaded_bytes": receipt["downloaded_bytes"], "output": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
