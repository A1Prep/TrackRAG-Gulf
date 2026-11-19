"""Task 1 dataset preparation on small synthetic archives in the EMT release layout."""

import io
import json
import tarfile
from pathlib import Path

import pytest
from PIL import Image

from tools import vision_prepare_dataset as prep
from trackrag.vision.classes import EMT_CLASSES

SIZE = (64, 36)  # 16:9 like EMT; long_side=32 halves it, as 2560x1440 -> 1280x720


def _jpeg(color):
    buffer = io.BytesIO()
    Image.new("RGB", SIZE, color).save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def _add(archive, name, payload):
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def _row(frame, track, name, box):
    left, top, right, bottom = box
    return f"{frame:05d} {track} {name} 0.0 0 0.0 {left} {top} {right} {bottom} 0.0 0.0 0.0 0.0 0.0 0.0 0.0"


@pytest.fixture()
def release(tmp_path):
    recordings = {"video_train": ("train", 11), "video_excluded": ("train", 6), "video_dev": ("train", 6),
                  "video_test": ("test", 6)}
    images = tmp_path / "train_images.tar.gz"
    with tarfile.open(images, "w:gz") as archive:
        for recording, (split, frames) in recordings.items():
            for frame in range(1, frames + 1):
                _add(archive, f"images/{split}/images/{recording}/{frame:05d}.jpg", _jpeg((frame * 20, 0, 0)))
    rows = {
        "video_train": [
            _row(1, 1, "Car", (10, 10, 30, 20)),
            _row(1, 2, "AV", (0, 30, 64, 36)),
            _row(1, 3, "Pedestrian", (-1.0, 5, 3, 25)),        # clipped, kept
            _row(1, 4, "Bus", (40, 10, 40.5, 30)),             # 0.25 px wide after resize -> dropped
            _row(6, 1, "Car", (12, 10, 32, 20)),
            _row(11, 5, "Motorbike", (50, 12, 60, 30)),
            _row(2, 1, "Car", (11, 10, 31, 20)),               # frame 2 not sampled
        ],
        "video_excluded": [_row(1, 1, "Car", (10, 10, 30, 20))],
        "video_dev": [_row(1, 1, "Bus", (20, 4, 60, 34)), _row(6, 1, "Bus", (21, 4, 61, 34))],
        "video_test": [_row(1, 1, "Car", (10, 10, 30, 20))],
    }
    annotations = tmp_path / "train_annotation.tar.gz"
    with tarfile.open(annotations, "w:gz") as archive:
        for recording, lines in rows.items():
            split = recordings[recording][0]
            _add(archive, f"EMT/annotations/{split}/{recording}.txt", ("\n".join(lines) + "\n").encode())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"train": ["video_train", "video_excluded"], "dev": ["video_dev"],
                                    "test": ["video_test"]}))
    return images, annotations, manifest


def _run(release, out, **options):
    images, annotations, manifest = release
    return prep.prepare(images, annotations, out, manifest_path=manifest,
                        excluded={"video_excluded": "test exclusion"}, long_side=32,
                        images_sha256=None, sanity=2, **options)


def test_prepare_builds_coco_splits_with_scaled_boxes(release, tmp_path):
    out = tmp_path / "ds"
    summary = _run(release, out, pack=tmp_path / "ds.tar")
    train = json.loads((out / "train" / "_annotations.coco.json").read_text())
    valid = json.loads((out / "valid" / "_annotations.coco.json").read_text())

    assert [c["name"] for c in train["categories"]] == list(EMT_CLASSES)
    assert [c["id"] for c in train["categories"]] == list(range(1, 10))
    assert all(c["supercategory"] == "none" for c in train["categories"])
    # Every 5th frame (1, 6, 11) of the training recording only; excluded/test never appear.
    assert [(i["recording_id"], i["frame_id"]) for i in train["images"]] == [
        ("video_train", 1), ("video_train", 6), ("video_train", 11)]
    assert [(i["recording_id"], i["frame_id"]) for i in valid["images"]] == [("video_dev", 1), ("video_dev", 6)]
    assert all((i["width"], i["height"]) == (32, 18) for i in train["images"] + valid["images"])
    assert all(Image.open(out / "train" / i["file_name"]).size == (32, 18) for i in train["images"])

    first = [a for a in train["annotations"] if a["image_id"] == 1]
    car = next(a for a in first if a["category_id"] == EMT_CLASSES.index("Car") + 1)
    assert car["bbox"] == [5.0, 5.0, 10.0, 5.0] and car["area"] == 50.0 and car["iscrowd"] == 0
    pedestrian = next(a for a in first if a["category_id"] == 1)
    assert pedestrian["bbox"] == [0.0, 2.5, 1.5, 10.0]
    assert [a["id"] for a in train["annotations"]] == list(range(1, len(train["annotations"]) + 1))
    assert summary["counts"]["train"]["by_class"]["Car"] == 2
    assert summary["dropped_or_adjusted_label_rows"] == {
        "clipped to the image (kept)": 1, "non-EMT class AV": 1, "smaller than 1 px after resizing": 1}
    assert summary["sampled_images_skipped_without_labels"] == 0
    assert summary["recordings"]["video_train"]["frames_in_archive"] == 11
    assert "video_excluded" not in summary["recordings"] and "video_test" not in summary["recordings"]
    assert len(list((out / "sanity").iterdir())) == 2
    assert (tmp_path / "ds.tar.sha256").read_text().split()[0] == summary["pack"]["sha256"]
    assert not (tmp_path / "ds.partial").exists()


def test_prepare_refuses_test_recordings_and_bad_exclusions(release, tmp_path):
    images, annotations, manifest = release
    data = json.loads(manifest.read_text())
    data["dev"].append("video_test")
    data["test"] = []
    leaky = tmp_path / "leaky.json"
    leaky.write_text(json.dumps(data))
    # video_test images live under images/test: refused even if a manifest calls it dev.
    with pytest.raises(AssertionError):
        prep.prepare(images, annotations, tmp_path / "x", manifest_path=leaky, excluded={},
                     long_side=32, images_sha256=None, sanity=0)
    with pytest.raises(ValueError):
        prep.plan_roles({"train": ["a"], "dev": ["b"], "test": ["c"]}, {"zzz": "not a training recording"})
    with pytest.raises(AssertionError):
        prep.plan_roles({"train": ["a", "c"], "dev": ["b"], "test": ["c"]}, {})


def test_prepare_verifies_archive_hash_and_refuses_existing_output(release, tmp_path):
    images, annotations, manifest = release
    with pytest.raises(ValueError, match="sha256"):
        prep.prepare(images, annotations, tmp_path / "ds", manifest_path=manifest,
                     excluded={"video_excluded": "x"}, long_side=32, images_sha256="0" * 64, sanity=0)
    (tmp_path / "exists").mkdir()
    with pytest.raises(FileExistsError):
        prep.prepare(images, annotations, tmp_path / "exists", manifest_path=manifest, images_sha256=None)


def _relabel(release, tmp_path, recording, lines):
    images, annotations, manifest = release
    with tarfile.open(annotations, "r:gz") as archive:
        members = {m.name: archive.extractfile(m).read() for m in archive if m.isfile()}
    members[f"EMT/annotations/train/{recording}.txt"] = ("\n".join(lines) + "\n").encode()
    changed = tmp_path / "changed_annotation.tar.gz"
    with tarfile.open(changed, "w:gz") as archive:
        for name, payload in members.items():
            _add(archive, name, payload)
    return images, changed, manifest


def test_labelled_frames_without_images_fail_like_a_truncated_archive(release, tmp_path):
    images, annotations, manifest = _relabel(release, tmp_path, "video_dev", [
        _row(1, 1, "Bus", (20, 4, 60, 34)), _row(9, 1, "Bus", (20, 4, 60, 34))])  # frame 9 has no image
    with pytest.raises(ValueError, match="labelled frames have no image"):
        prep.prepare(images, annotations, tmp_path / "ds", manifest_path=manifest,
                     excluded={"video_excluded": "x"}, long_side=32, images_sha256=None, sanity=0)
    assert not (tmp_path / "ds").exists() and not (tmp_path / "ds.partial").exists()


def test_sampled_frames_without_labels_are_skipped_not_trained_as_empty(release, tmp_path):
    images, annotations, manifest = _relabel(release, tmp_path, "video_dev", [_row(1, 1, "Bus", (20, 4, 60, 34))])
    summary = prep.prepare(images, annotations, tmp_path / "ds", manifest_path=manifest,
                           excluded={"video_excluded": "x"}, long_side=32, images_sha256=None, sanity=0)
    valid = json.loads((tmp_path / "ds" / "valid" / "_annotations.coco.json").read_text())
    assert [i["frame_id"] for i in valid["images"]] == [1]  # frame 6 had an image but no labels
    assert summary["sampled_images_skipped_without_labels"] == 1
    assert summary["recordings"]["video_dev"]["images_without_labels"] == 5


def test_boxes_far_outside_the_image_fail(release, tmp_path):
    images, annotations, manifest = _relabel(release, tmp_path, "video_dev", [
        _row(1, 1, "Bus", (20, 4, 90, 34)), _row(6, 1, "Bus", (21, 4, 61, 34))])  # right=90 on a 64 px image
    with pytest.raises(ValueError, match="outside"):
        prep.prepare(images, annotations, tmp_path / "ds", manifest_path=manifest,
                     excluded={"video_excluded": "x"}, long_side=32, images_sha256=None, sanity=0)


def test_existing_pack_target_is_refused_before_any_work(release, tmp_path):
    (tmp_path / "ds.tar.sha256").write_text("old")
    with pytest.raises(FileExistsError, match="Pack target"):
        _run(release, tmp_path / "ds", pack=tmp_path / "ds.tar")
    assert not (tmp_path / "ds").exists()


def test_default_exclusions_and_real_manifest_roles():
    manifest = prep.load_manifest(Path(__file__).resolve().parents[1] / "splits" / "task2_manifest.json")
    roles = prep.plan_roles(manifest, prep.EXCLUDED_FROM_TRAINING)
    training = sorted(r for r, role in roles.items() if role == "train")
    assert len(training) == 11
    assert "video_054604" not in roles and "video_204347" not in roles
    assert sorted(r for r, role in roles.items() if role == "valid") == ["video_054907", "video_174340"]
    assert not set(roles) & set(manifest["test"])
