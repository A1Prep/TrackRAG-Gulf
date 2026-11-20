"""Small generated fixtures for the bounded downloader, not model-quality tests."""

from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile
import zipfile

from PIL import Image
import pytest

from tools.dropoff_bootstrap_regseg import (
    BootstrapError, BoundedSource, build_cases, excluded_source_ids,
    mask_targets, read_outer_member, safe_name, tar_files, zip_directory,
    verified_scene_map,
)


class Response(io.BytesIO):
    def __init__(self, body, *, status=206, content_range="bytes 0-3/4", encoding=None):
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Range": content_range}
        if encoding:
            self.headers["Content-Encoding"] = encoding


def image_bytes(*, mask=False, empty=False):
    if mask:
        image = Image.new("L", (8, 4), 0)
        if not empty:
            # A U shape: the bounding-box centroid is background.
            for y in range(4):
                image.putpixel((0, y), 255)
                image.putpixel((7, y), 255)
            for x in range(8):
                image.putpixel((x, 3), 255)
    else:
        image = Image.new("RGB", (16, 8), "white")
    buffer = io.BytesIO()
    image.save(buffer, "PNG" if mask else "JPEG")
    return buffer.getvalue()


def tar_bytes(files):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, content in files:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return buffer.getvalue()


def test_download_rejects_budget_before_contacting_source():
    def forbidden(*args, **kwargs):
        pytest.fail("Over-budget source should never be contacted")
    with pytest.raises(BootstrapError, match="budget"):
        BoundedSource(3, opener=forbidden).fetch("https://example.test/data", 0, 3, total=4)


@pytest.mark.parametrize("response", [
    lambda: Response(b"data", status=200),
    lambda: Response(b"data", content_range="bytes 1-4/5"),
    lambda: Response(b"data", content_range="bytes 0-3/5"),
    lambda: Response(b"dat"),
    lambda: Response(b"data!"),
    lambda: Response(b"data", encoding="gzip"),
])
def test_download_requires_exact_identity_and_body(response):
    source = BoundedSource(10, opener=lambda *args, **kwargs: response())
    with pytest.raises(BootstrapError):
        source.fetch("https://example.test/data", 0, 3, total=4)


def test_download_receipt_and_cumulative_limit():
    calls = []
    def opener(request, timeout):
        calls.append(request.get_header("Range"))
        return Response(b"data")
    source = BoundedSource(7, opener=opener)
    assert source.fetch("https://example.test/data", 0, 3, total=4) == b"data"
    assert calls == ["bytes=0-3"]
    assert source.used_bytes == 4
    assert source.receipts[0]["source_size"] == 4
    assert len(source.receipts[0]["sha256"]) == 64
    with pytest.raises(BootstrapError, match="budget"):
        source.fetch("https://example.test/data", 0, 3, total=4)


def test_pinned_small_file_rejects_changed_content():
    source = BoundedSource(10, opener=lambda *args, **kwargs: Response(b"data"))
    with pytest.raises(BootstrapError, match="SHA256"):
        source.small_file("https://example.test/data", 4, "0" * 64)


@pytest.mark.parametrize("name", ["../x", "/x", "a/../../x", "a\\x", "C:/x", ""])
def test_archive_paths_cannot_escape(name):
    with pytest.raises(BootstrapError, match="Unsafe"):
        safe_name(name)


def test_central_directory_reader_matches_zipfile():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("release/images.tar", b"fixture-images")
        archive.writestr("release/masks.tar", b"fixture-masks")
    content = buffer.getvalue()
    parsed = zip_directory(content, len(content))
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for member in archive.infolist():
            assert parsed[member.filename] == {
                "offset": member.header_offset, "compressed_size": member.compress_size,
                "size": member.file_size, "crc32": member.CRC, "compression": member.compress_type,
            }
    with pytest.raises(BootstrapError):
        zip_directory(content[:-1], len(content) - 1)


def test_outer_member_is_range_bounded_crc_checked_and_size_limited(monkeypatch):
    import tools.dropoff_bootstrap_regseg as module
    payload = b"actual fixture TAR bytes" * 100
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("release/images.tar", payload)
    blob = buffer.getvalue()
    entry = zip_directory(blob, len(blob))["release/images.tar"]
    monkeypatch.setattr(module, "ARCHIVE_SIZE", len(blob))
    requests = []
    def opener(request, timeout):
        import re
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range"))
        start, end = int(match[1]), int(match[2])
        requests.append((start, end))
        return Response(blob[start:end + 1], content_range=f"bytes {start}-{end}/{len(blob)}")
    source = BoundedSource(10000, opener=opener)
    decoded, receipt = read_outer_member(source, "release/images.tar", entry)
    assert decoded == payload
    assert receipt["complete"] is True
    assert receipt["compressed_bytes_read"] == entry["compressed_size"]
    assert sum(end - start + 1 for start, end in requests) < len(blob)
    with pytest.raises(BootstrapError, match="safety limit"):
        read_outer_member(BoundedSource(10000, opener=opener), "release/images.tar", entry, max_uncompressed=10)
    prefix, partial = read_outer_member(
        BoundedSource(10000, opener=opener), "release/images.tar", entry, prefix_bytes=2,
    )
    assert len(prefix) < len(payload)
    assert partial["complete"] is False
    assert partial["compressed_bytes_read"] == 2


def test_outer_member_refuses_local_directory_identity_mismatch(monkeypatch):
    import tools.dropoff_bootstrap_regseg as module
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("release/images.tar", b"fixture")
    original = buffer.getvalue()
    entry = zip_directory(original, len(original))["release/images.tar"]
    corrupted = bytearray(original)
    corrupted[30] = ord("X")
    blob = bytes(corrupted)
    monkeypatch.setattr(module, "ARCHIVE_SIZE", len(blob))
    def opener(request, timeout):
        import re
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range"))
        start, end = int(match[1]), int(match[2])
        return Response(blob[start:end + 1], content_range=f"bytes {start}-{end}/{len(blob)}")
    with pytest.raises(BootstrapError, match="identity"):
        read_outer_member(BoundedSource(10000, opener=opener), "release/images.tar", entry)


def test_tar_only_returns_matching_training_members():
    content = tar_bytes([
        ("imgs/img_train_7.jpg", b"train"),
        ("imgs/img_val_7.jpg", b"validation"),
        ("imgs/img_test_7.jpg", b"test"),
    ])
    assert tar_files(content, r"imgs/img_train_(\d+)\.jpg") == {7: b"train"}
    with pytest.raises(BootstrapError):
        tar_files(tar_bytes([("../bad", b"bad")]), r"imgs/img_train_(\d+)\.jpg")


def test_partial_tar_retains_only_fully_read_members():
    content = tar_bytes([
        ("imgs/img_train_1.jpg", b"a" * 700),
        ("imgs/img_train_2.jpg", b"b" * 700),
    ])
    # First header+content+padding, then second header and incomplete body.
    truncated = content[:1536 + 512 + 100]
    assert tar_files(truncated, r"imgs/img_train_(\d+)\.jpg", partial=True) == {1: b"a" * 700}
    with pytest.raises(BootstrapError):
        tar_files(truncated, r"imgs/img_train_(\d+)\.jpg", partial=False)


def test_concise_test_source_ids_are_excluded_and_unique():
    assert excluded_source_ids(b"4 a rewritten command\n9 another one\n") == {4, 9}
    with pytest.raises(BootstrapError):
        excluded_source_ids(b"4 first\n4 second\n")
    with pytest.raises(BootstrapError):
        excluded_source_ids(b"unknown command\n")


def scene_sources():
    metadata = {"train": {
        "1": {"img": "img_train_1.jpg", "command": "one", "command_token": "a"},
        "2": {"img": "img_train_2.jpg", "command": "two", "command_token": "b"},
        "3": {"img": "img_train_3.jpg", "command": "three", "command_token": "c"},
    }}
    source = {"commands": [
        {"t2c_img": row["img"], "command": row["command"], "command_token": row["command_token"],
         "scene_token": ("1" if identity in ("1", "2") else "2") * 32,
         "sample_token": identity * 32}
        for identity, row in metadata["train"].items()
    ]}
    return metadata, source


def test_scene_mapping_excludes_adjacent_scene_images_not_only_exact_ids(tmp_path):
    metadata, original = scene_sources()
    scenes, excluded = verified_scene_map(metadata, original, {1})
    assert excluded == {"1" * 32}
    images = {identity: image_bytes() for identity in (1, 2, 3)}
    masks = {identity: image_bytes(mask=True) for identity in (1, 2, 3)}
    cases = build_cases(
        metadata, images, masks, {1}, tmp_path / "scene-pilot", limit=10,
        source_receipt={"fixture": True}, scene_map=scenes, excluded_scenes=excluded,
    )
    assert len(cases) == 1
    assert cases[0]["source"]["source_index"] == 3
    assert cases[0]["source"]["scene_group_verified"] is True
    assert cases[0]["source"]["scene_token"] == "2" * 32
    assert cases[0]["scene_group"] == "talk2car-scene:" + "2" * 32


@pytest.mark.parametrize("field,value", [
    ("t2c_img", "img_val_1.jpg"), ("command", "changed command"),
    ("command_token", "foreign"), ("scene_token", "not-a-source-token"),
])
def test_scene_mapping_requires_token_image_text_and_valid_scene(field, value):
    metadata, source = scene_sources()
    source["commands"][0][field] = value
    with pytest.raises(BootstrapError):
        verified_scene_map(metadata, source, {1})


def test_scene_mapping_refuses_missing_exclusion_or_duplicate_source_token():
    metadata, source = scene_sources()
    with pytest.raises(BootstrapError, match="absent"):
        verified_scene_map(metadata, source, {999})
    source["commands"].append(source["commands"][0])
    with pytest.raises(BootstrapError, match="Duplicate"):
        verified_scene_map(metadata, source, {1})


def test_nonconvex_mask_target_is_inside_and_maps_to_original_pixels():
    shape, point, box, mask_shape = mask_targets(image_bytes(), image_bytes(mask=True))
    assert shape == (16, 8)
    assert mask_shape == (8, 4)
    assert box == [0.0, 0.0, 16.0, 8.0]
    with Image.open(io.BytesIO(image_bytes(mask=True))) as mask:
        assert mask.getpixel((int(point[0] / 2), int(point[1] / 2))) > 0
        assert mask.getpixel((3, 1)) == 0  # A naive central target would fail.
    with pytest.raises(BootstrapError, match="Empty"):
        mask_targets(image_bytes(), image_bytes(mask=True, empty=True))


def make_metadata(*ids):
    return {"train": {str(i): {"img": f"img_train_{i}.jpg", "command": f"fixture command {i}"} for i in ids}}


def test_case_export_preserves_partial_supervision_and_portable_paths(tmp_path):
    output = tmp_path / "pilot"
    cases = build_cases(
        make_metadata(7, 9), {7: image_bytes(), 9: image_bytes()},
        {7: image_bytes(mask=True), 9: image_bytes(mask=True)}, {9}, output,
        limit=10, source_receipt={"fixture": True},
    )
    assert len(cases) == 1
    case = cases[0]
    assert case["partition"] == "train"
    assert case["source"]["source_index"] == 7
    assert case["source"]["scene_group_verified"] is False
    assert case["request"]["scene"]["provenance"]["detector_run"] is False
    assert set(case["target"]) == {"destination_point_xy", "destination_box_xyxy"}
    assert set(case["supervised_fields"]) == set(case["target"])
    assert "status" not in case["target"]
    assert "target" not in case["request"]
    assert case["request"]["scene"]["frame"]["image_path"] == "images/img_train_7.jpg"
    assert case["source"]["mask_path"] == "masks/gt_img_ann_train_7.png"
    assert (output / case["source"]["mask_path"]).read_bytes() == image_bytes(mask=True)
    assert json.loads((output / "cases.jsonl").read_text()) == case
    with pytest.raises(BootstrapError, match="overwrite"):
        build_cases(make_metadata(7), {}, {}, set(), output, limit=1, source_receipt={})


def test_case_export_refuses_test_derived_or_mismatched_records(tmp_path):
    images, masks = {7: image_bytes()}, {7: image_bytes(mask=True)}
    with pytest.raises(BootstrapError, match="No safe"):
        build_cases(make_metadata(7), images, masks, {7}, tmp_path / "excluded", limit=1, source_receipt={})
    metadata = make_metadata(7)
    metadata["train"]["7"]["img"] = "img_val_7.jpg"
    with pytest.raises(BootstrapError, match="identity"):
        build_cases(metadata, images, masks, set(), tmp_path / "wrong", limit=1, source_receipt={})
    assert not (tmp_path / "wrong").exists()
