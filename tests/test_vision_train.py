"""Training-wrapper logic that runs without rfdetr/torch: subsets, dataset checks, Drive sync, restore."""

import json
import os
import time

import pytest

from tools import vision_train as vt
from trackrag.vision.classes import EMT_CLASSES


def _dataset(root, train_images=3, valid_images=2, categories=None):
    categories = categories or [{"id": i + 1, "name": n, "supercategory": "none"} for i, n in enumerate(EMT_CLASSES)]
    for split, count in (("train", train_images), ("valid", valid_images)):
        (root / split).mkdir(parents=True)
        images = []
        for index in range(1, count + 1):
            name = f"rec_{index:05d}.jpg"
            (root / split / name).write_bytes(b"jpeg")
            images.append({"id": index, "file_name": name, "width": 32, "height": 18})
        annotations = [{"id": i, "image_id": i, "category_id": 4, "bbox": [1, 1, 4, 4], "area": 16, "iscrowd": 0}
                       for i in range(1, count + 1)]
        (root / split / "_annotations.coco.json").write_text(json.dumps(
            {"images": images, "annotations": annotations, "categories": categories}))
    return root


def test_make_subset_and_check_dataset(tmp_path):
    dataset = _dataset(tmp_path / "ds")
    subset = vt.make_subset(dataset, tmp_path / "micro", 2, 1)
    assert vt.check_dataset(subset) == {"train": {"images": 2, "annotations": 2},
                                         "valid": {"images": 1, "annotations": 1}}
    assert sorted(p.name for p in (subset / "train").iterdir()) == [
        "_annotations.coco.json", "rec_00001.jpg", "rec_00002.jpg"]


def test_check_dataset_rejects_wrong_class_order_and_missing_images(tmp_path):
    reordered = [{"id": i + 1, "name": n, "supercategory": "none"} for i, n in enumerate(reversed(EMT_CLASSES))]
    with pytest.raises(ValueError):
        vt.check_dataset(_dataset(tmp_path / "a", categories=reordered))
    broken = _dataset(tmp_path / "b")
    (broken / "train" / "rec_00001.jpg").unlink()
    with pytest.raises(FileNotFoundError):
        vt.check_dataset(broken)


def test_drive_sync_copies_resumable_and_small_files_only(tmp_path):
    source, target = tmp_path / "run", tmp_path / "drive"
    source.mkdir()
    (source / "last.ckpt").write_bytes(b"state-1")
    (source / "checkpoint_best_ema.pth").write_bytes(b"best")
    (source / "metrics.csv").write_text("epoch,map\n0,0.1\n")
    (source / "checkpoint_3.ckpt").write_bytes(b"archive")        # interval archive: not synced
    (source / "other_weights.pth").write_bytes(b"w")              # unnamed weights: not synced
    (source / "logs").mkdir()
    (source / "logs" / "events.out").write_bytes(b"tb")
    sync = vt.DriveSync(source, target, every_seconds=3600)
    assert sync.sync_once() == 4
    assert sorted(p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()) == [
        "checkpoint_best_ema.pth", "last.ckpt", "logs/events.out", "metrics.csv"]
    assert sync.sync_once() == 0  # unchanged files are not copied again
    time.sleep(0.01)
    (source / "last.ckpt").write_bytes(b"state-2")
    os.utime(source / "last.ckpt")
    assert sync.sync_once() == 1
    assert (target / "last.ckpt").read_bytes() == b"state-2"
    assert not list(target.rglob("*.tmp-sync"))


def test_drive_sync_stop_waits_for_the_background_thread_and_logs_errors(tmp_path, monkeypatch, capsys):
    source, target = tmp_path / "run", tmp_path / "drive"
    source.mkdir()
    (source / "last.ckpt").write_bytes(b"state")
    sync = vt.DriveSync(source, target, every_seconds=0.01).start()
    time.sleep(0.2)
    assert sync.stop() == 0 and sync.copies == 1  # copied once by the thread, nothing left for stop()
    assert not sync._thread.is_alive()
    monkeypatch.setattr(vt.shutil, "copy2", lambda *a, **k: (_ for _ in ()).throw(OSError(107, "not connected")))
    (source / "metrics.csv").write_text("x")
    assert sync.sync_once() == 0
    assert "ERROR metrics.csv" in capsys.readouterr().out and sync.errors


def test_drive_sync_refuses_an_unmounted_drive_path(monkeypatch):
    monkeypatch.setattr(vt.os.path, "ismount", lambda path: False)
    with pytest.raises(RuntimeError, match="not mounted"):
        vt.require_mounted(vt.Path("/content/drive/MyDrive/task1"))
    vt.require_mounted(vt.Path("/tmp/elsewhere"))  # non-Drive paths are fine


def test_restore_prefers_a_newer_local_checkpoint(tmp_path):
    sync_dir, output = tmp_path / "drive", tmp_path / "run"
    sync_dir.mkdir()
    output.mkdir()
    (sync_dir / "last.ckpt").write_bytes(b"older-on-drive")
    os.utime(sync_dir / "last.ckpt", (1_000_000, 1_000_000))
    (output / "last.ckpt").write_bytes(b"newer-local")
    assert vt.restore(sync_dir, output).read_bytes() == b"newer-local"


def test_weights_need_a_folder():
    with pytest.raises(ValueError, match="folder"):
        vt.resolve_weights("rf-detr-large-2026.pth")
    assert vt.resolve_weights("models/rf-detr-large-2026.pth").name == "rf-detr-large-2026.pth"


def test_restore_requires_last_ckpt_and_copies_state(tmp_path):
    sync_dir, output = tmp_path / "drive", tmp_path / "run"
    sync_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        vt.restore(sync_dir, output)
    (sync_dir / "last.ckpt").write_bytes(b"state")
    (sync_dir / "metrics.csv").write_text("x")
    resume = vt.restore(sync_dir, output)
    assert resume == output / "last.ckpt" and resume.read_bytes() == b"state"
    assert (output / "metrics.csv").read_text() == "x"


def test_base_weights_must_be_the_2026_checkpoint(tmp_path):
    wrong_name = tmp_path / "rf-detr-large.pth"
    wrong_name.write_bytes(b"x")
    with pytest.raises(ValueError, match="2026"):
        vt.verify_base_weights(wrong_name)
    wrong_hash = tmp_path / "rf-detr-large-2026.pth"
    wrong_hash.write_bytes(b"not the real checkpoint")
    with pytest.raises(ValueError, match="md5"):
        vt.verify_base_weights(wrong_hash)


def test_train_options_are_real_rfdetr_trainconfig_fields(tmp_path, monkeypatch):
    """Runs where rfdetr is installed (model env); skipped in CI. Catches renamed or misplaced options."""
    config = pytest.importorskip("rfdetr.config")
    seen = {}
    monkeypatch.setattr(vt, "train", lambda args: seen.setdefault("args", args))
    vt.main(["--dataset", "d", "--output", "o", "--weights", "w", "--epochs", "3", "--batch-size", "4"])
    seen["args"].device = "cuda"
    options = vt.build_train_options(seen["args"], tmp_path / "ds", tmp_path / "out", tmp_path / "last.ckpt")
    unknown = sorted(set(options) - set(config.TrainConfig.model_fields))
    assert unknown == ["device"]  # popped by rfdetr before TrainConfig and mapped to the accelerator
    import inspect
    import rfdetr.detr
    assert 'kwargs.pop("device", None)' in inspect.getsource(rfdetr.detr)
    assert "resolution" in config.RFDETRLargeConfig.model_fields  # model-level, passed to the constructor
    assert options["checkpoint_interval"] > options["epochs"]
    config.TrainConfig(**{k: v for k, v in options.items() if k != "device"})  # pydantic validation


def test_cli_defaults_grad_accum_and_validates_resolution(monkeypatch):
    seen = {}
    monkeypatch.setattr(vt, "train", lambda args: seen.setdefault("args", args))
    vt.main(["--dataset", "d", "--output", "o", "--weights", "w", "--epochs", "3", "--batch-size", "4"])
    assert seen["args"].grad_accum == 4 and seen["args"].seed == 42
    assert seen["args"].augmentation_backend == "torchvision"
    with pytest.raises(SystemExit):
        vt.main(["--dataset", "d", "--output", "o", "--weights", "w", "--epochs", "3", "--resolution", "700"])
