"""Task 2 tests for the EMT recording reader.

Run from the repository root:

    pytest tests/test_emt.py -v
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from trackrag.emt import (
    DEFAULT_FRAME_RATE,
    FrameNotFoundError,
    MissingLabelsError,
    assert_splits_disjoint,
    load_annotated_tracks,
    make_split_manifest,
    open_recording,
    read_release_frame_rate,
    recording_ids_for_split,
)


def _write_frame(root: Path, recording_id: str, frame_id: int, split: str = "train", color=(30, 60, 90)) -> None:
    path = root / "images" / split / "images" / recording_id / f"{frame_id:05d}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((90, 160, 3), dtype=np.uint8)
    image[:] = color
    cv2.imwrite(str(path), image)


@pytest.fixture()
def emt_root(tmp_path: Path) -> Path:
    root = tmp_path / "emt"
    for split in ("train", "test"):
        (root / "images" / split / "images").mkdir(parents=True, exist_ok=True)
        (root / "annotations" / split).mkdir(parents=True, exist_ok=True)
    for frame_id in (3, 1, 5, 2):
        _write_frame(root, "rec_a", frame_id)
    _write_frame(root, "rec_b", 1, color=(40, 90, 50))
    _write_frame(root, "rec_images_only", 1)
    _write_frame(root, "rec_official_test", 1, split="test")
    (root / "annotations" / "train" / "rec_a.txt").write_text(
        "00001 7 Car 0.0 0 0.0 20.0 15.0 70.0 70.0\n"
        "00002 7 Car 0.0 0 0.0 24.0 16.0 74.0 71.0\n"
        "00004 99 Car 0.0 0 0.0 10.0 10.0 30.0 30.0\n"
        "00005 8 Pedestrian 0.0 0 0.0 90.0 25.0 120.0 85.0\n"
        "00001 50 AV 0.0 0 0.0 0.0 0.0 10.0 10.0\n"
    )
    (root / "annotations" / "train" / "rec_b.txt").write_text(
        "00001 7 Bus 0.0 0 0.0 30.0 20.0 100.0 80.0\n"
    )
    return root


def test_iter_frames_numerical_order(emt_root):
    rec = open_recording(emt_root, "rec_a", frame_rate=10.0)
    assert rec.split == "train"
    assert rec.frame_ids == [1, 2, 3, 5]
    assert [f["frame_id"] for f in rec.iter_frames()] == [1, 2, 3, 5]


def test_frame_contract_and_exact_path(emt_root):
    rec = open_recording(emt_root, "rec_a", frame_rate=10.0)
    frame = rec.get_frame(5)
    assert set(frame) == {"recording_id", "frame_id", "timestamp_s", "width", "height", "image_path"}
    assert frame["image_path"].endswith("00005.jpg")
    assert (frame["width"], frame["height"]) == (160, 90)
    assert abs(frame["timestamp_s"] - 0.4) < 1e-12


def test_clock_starts_at_first_image(emt_root):
    rec = open_recording(emt_root, "rec_a", frame_rate=10.0)
    assert rec.get_frame(rec.first_frame_id)["timestamp_s"] == 0.0


def test_missing_frame_raises_keyerror(emt_root):
    rec = open_recording(emt_root, "rec_a")
    with pytest.raises(FrameNotFoundError):
        rec.get_frame(4)
    with pytest.raises(KeyError):
        rec.get_frame(4)


def test_annotations_use_original_boxes(emt_root):
    rec = open_recording(emt_root, "rec_a")
    obs = load_annotated_tracks(rec)
    assert [o["frame_id"] for o in obs] == [1, 2, 5]
    assert obs[0]["bbox_xyxy"] == (20.0, 15.0, 70.0, 70.0)
    assert all(o["source"] == "annotated" for o in obs)
    assert all(o["recording_id"] == "rec_a" for o in obs)
    assert [o for o in obs if o["frame_id"] == 3] == []
    assert [o for o in obs if o["class_name"] == "AV"] == []


def test_images_only_mode(emt_root):
    rec = open_recording(emt_root, "rec_images_only")
    assert cv2.imread(rec.get_frame(1)["image_path"]).shape == (90, 160, 3)
    with pytest.raises(MissingLabelsError):
        load_annotated_tracks(rec)


def test_track_ids_are_recording_local(emt_root):
    obs_a = load_annotated_tracks(open_recording(emt_root, "rec_a"))
    obs_b = load_annotated_tracks(open_recording(emt_root, "rec_b"))
    assert obs_a[0]["track_id"] == obs_b[0]["track_id"] == 7
    assert {o["recording_id"] for o in obs_b} == {"rec_b"}
    assert obs_b[0]["class_name"] == "Bus"


def test_manifest_is_disjoint_and_preserves_official_test(emt_root, tmp_path):
    manifest = make_split_manifest(emt_root, dev_count=1, seed=42, output_path=tmp_path / "manifest.json")
    available = set(recording_ids_for_split(emt_root, "train")) | set(recording_ids_for_split(emt_root, "test"))
    assert_splits_disjoint(manifest, available)
    assert manifest["test"] == ["rec_official_test"]
    assert set(manifest["train"]) | set(manifest["dev"]) | set(manifest["test"]) == available
    assert set(manifest["train"]) | set(manifest["dev"]) == set(recording_ids_for_split(emt_root, "train"))


def test_duplicate_split_detected():
    with pytest.raises(ValueError):
        assert_splits_disjoint({"train": ["x"], "dev": ["x"], "test": []})


def test_frame_rate_validation(emt_root):
    with pytest.raises(ValueError):
        open_recording(emt_root, "rec_a", frame_rate=0)


def test_release_frame_rate_missing_returns_none(emt_root):
    assert read_release_frame_rate(emt_root) is None


def test_default_frame_rate_is_used_when_metadata_absent(emt_root):
    rec = open_recording(emt_root, "rec_a")
    assert rec.frame_rate == DEFAULT_FRAME_RATE


@pytest.mark.parametrize("frame_rate", [float("nan"), float("inf")])
def test_nonfinite_frame_rate_is_rejected(emt_root, frame_rate):
    with pytest.raises(ValueError):
        open_recording(emt_root, "rec_a", frame_rate=frame_rate)


@pytest.mark.parametrize("row", [
    "00001.5 7 Car 0 0 0 20 15 70 70\n",
    "00001 7.5 Car 0 0 0 20 15 70 70\n",
    "00001 7 Car 0 0 0 nan 15 70 70\n",
])
def test_invalid_annotation_numbers_are_rejected(emt_root, row):
    labels = emt_root / "annotations" / "train" / "rec_a.txt"
    labels.write_text(row)
    with pytest.raises(ValueError):
        load_annotated_tracks(open_recording(emt_root, "rec_a"))


def test_duplicate_track_in_one_frame_is_rejected(emt_root):
    labels = emt_root / "annotations" / "train" / "rec_a.txt"
    labels.write_text("00001 7 Car 0 0 0 20 15 70 70\n" * 2)
    with pytest.raises(ValueError):
        load_annotated_tracks(open_recording(emt_root, "rec_a"))


@pytest.mark.parametrize("value", ["NaN", "inf", "0", "-5"])
def test_invalid_declared_frame_rate_does_not_fall_back_to_default(emt_root, value):
    (emt_root / "annotations" / "metadata.txt").write_text(f"fps={value}\n")
    with pytest.raises(ValueError):
        open_recording(emt_root, "rec_a")


def test_valid_declared_frame_rate_controls_timestamps(emt_root):
    (emt_root / "annotations" / "metadata.txt").write_text("fps=20\n")
    recording = open_recording(emt_root, "rec_a")
    assert recording.frame_rate == 20
    assert recording.get_frame(5)["timestamp_s"] == pytest.approx(0.2)
