"""Fine-tune RF-DETR Large (2026 checkpoint) on the Task 1 EMT dataset, with Drive sync and resume.

Typical Colab use (see notebooks/task1_colab.ipynb):

    # 1) micro run: proves training, Drive sync, reload and class order in minutes
    python -m tools.vision_train --dataset /content/ds_v1 --output /content/runs/micro \
        --weights /content/models/rf-detr-large-2026.pth --micro 100 40 --epochs 1 --batch-size auto \
        --device cuda --sync-dir /content/drive/MyDrive/task1/runs/micro --sync-every-min 1

    # 2) full run, syncing resumable state to Drive every 10 minutes
    python -m tools.vision_train --dataset /content/ds_v1 --output /content/runs/ft01 \
        --weights /content/models/rf-detr-large-2026.pth --epochs 12 --batch-size 8 --device cuda \
        --sync-dir /content/drive/MyDrive/task1/runs/ft01

    # 3) after a disconnect: same command plus --resume (restores the Drive state when it is newer)

Settings: checkpoint_interval larger than the epoch count so rfdetr
keeps a rolling ``last.ckpt`` (interval 1 disables it in rfdetr 1.12.1), torchvision
augmentation backend (antialiased like validation), fixed seed, no early stopping, no test
pass, per-class validation metrics logged. ``checkpoint_best_total.pth`` is written at a
normal finish (EMA weights of the best validation epoch). If training stops early, the run can
be resumed; a ``task1_training_failed_*.json`` records what happened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import threading
import time
import uuid
from importlib import metadata
from pathlib import Path

from trackrag.vision.classes import EMT_CLASSES, require_emt_class_order

LARGE_2026 = {"file": "rf-detr-large-2026.pth", "md5": "5cb72153541cbcb9aa6efa26222acc75"}
# Copy order matters: last.ckpt first, so Drive never holds a best file newer than its resume state.
SYNC_ORDER = ("last.ckpt", "checkpoint_best_ema.pth", "checkpoint_best_total.pth", "last_ema.pth",
              "metrics.csv", "training_config.json")
SMALL_FILE_BYTES = 50 * 1024 * 1024
DRIVE_ROOT = "/content/drive"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def versions() -> dict:
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("torch", "torchvision", "rfdetr", "pytorch_lightning", "transformers", "supervision",
                    "albumentations", "numpy", "pillow", "ultralytics"):
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = None
    try:
        import torch

        result["cuda_available"] = bool(torch.cuda.is_available())
        result["cuda"] = torch.version.cuda
        result["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as error:  # pragma: no cover - environment report only
        result["torch_error"] = repr(error)
    return result


def make_subset(dataset: Path, target: Path, train_images: int, valid_images: int) -> Path:
    """Copy the first N images per split (and their annotations) for the micro run."""
    if target.exists():
        shutil.rmtree(target)
    for split, limit in (("train", train_images), ("valid", valid_images)):
        source = json.loads((dataset / split / "_annotations.coco.json").read_text(encoding="utf-8"))
        images = source["images"][:limit]
        keep = {image["id"] for image in images}
        (target / split).mkdir(parents=True)
        for image in images:
            shutil.copy2(dataset / split / image["file_name"], target / split / image["file_name"])
        subset = dict(source, images=images,
                      annotations=[a for a in source["annotations"] if a["image_id"] in keep])
        (target / split / "_annotations.coco.json").write_text(json.dumps(subset), encoding="utf-8")
    return target


def check_dataset(dataset: Path) -> dict:
    counts = {}
    for split in ("train", "valid"):
        data = json.loads((dataset / split / "_annotations.coco.json").read_text(encoding="utf-8"))
        names = [c["name"] for c in sorted(data["categories"], key=lambda c: c["id"])]
        require_emt_class_order(names)
        if [c["id"] for c in sorted(data["categories"], key=lambda c: c["id"])] != list(range(1, 10)):
            raise ValueError(f"{split}: category ids must be 1..9")
        missing = [i["file_name"] for i in data["images"] if not (dataset / split / i["file_name"]).is_file()]
        if missing:
            raise FileNotFoundError(f"{split}: {len(missing)} images missing, e.g. {missing[:3]}")
        counts[split] = {"images": len(data["images"]), "annotations": len(data["annotations"])}
    return counts


def require_mounted(target: Path) -> None:
    """Refuse to 'sync' into a local folder that merely looks like Google Drive."""
    if Path(target).as_posix().startswith(DRIVE_ROOT) and not os.path.ismount(DRIVE_ROOT):
        raise RuntimeError(f"{DRIVE_ROOT} is not mounted: mount Google Drive before training")


class DriveSync:
    """Copy resumable state and small artefacts to a mounted Drive folder.

    Each copy goes to a unique temporary name and is renamed into place only if the source did
    not change during the copy. A lock serialises the background thread and the final sync.
    Every copy and every error is logged immediately.
    """

    def __init__(self, source: Path, target: Path, every_seconds: float):
        self.source, self.target, self.every = Path(source), Path(target), every_seconds
        require_mounted(self.target)
        self.target.mkdir(parents=True, exist_ok=True)
        probe = self.target / f".sync-probe-{uuid.uuid4().hex[:8]}"
        probe.write_text("ok", encoding="utf-8")  # fails loudly if the target is not writable
        probe.unlink()
        self._seen: dict[str, tuple[int, int]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.errors: list[str] = []
        self.copies = 0

    def _candidates(self) -> list[Path]:
        """Named resumable/result files (in SYNC_ORDER) plus small logs; never other weights."""
        if not self.source.exists():
            return []
        named = [self.source / name for name in SYNC_ORDER if (self.source / name).is_file()]
        others = []
        for path in sorted(self.source.rglob("*")):
            if (not path.is_file() or ".tmp-sync" in path.name or path.name in SYNC_ORDER
                    or path.suffix in {".ckpt", ".pth"}):
                continue
            if path.stat().st_size <= SMALL_FILE_BYTES:
                others.append(path)
        return named + others

    def mark_synced(self) -> None:
        """Record current local files as already on Drive (used right after a restore)."""
        with self._lock:
            for path in self._candidates():
                stat = path.stat()
                self._seen[path.relative_to(self.source).as_posix()] = (stat.st_size, stat.st_mtime_ns)

    def sync_once(self) -> int:
        copied = 0
        with self._lock:
            for path in self._candidates():
                temporary = None
                key = path.relative_to(self.source).as_posix()
                try:
                    before = path.stat()
                    signature = (before.st_size, before.st_mtime_ns)
                    if self._seen.get(key) == signature:
                        continue
                    destination = self.target / key
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_name(
                        f"{destination.name}.{os.getpid()}-{uuid.uuid4().hex[:8]}.tmp-sync")
                    started = time.time()
                    shutil.copy2(path, temporary)
                    after = path.stat()
                    if (after.st_size, after.st_mtime_ns) != signature:
                        temporary.unlink()  # being rewritten right now; retry next round
                        continue
                    temporary.replace(destination)
                    temporary = None
                    self._seen[key] = signature
                    copied += 1
                    self.copies += 1
                    if path.name in SYNC_ORDER:
                        log(f"[sync] {key} -> Drive ({before.st_size / 1e6:.0f} MB, {time.time() - started:.0f}s)")
                except OSError as error:
                    message = f"[sync] ERROR {key}: {type(error).__name__}: {error}"
                    self.errors.append(message)
                    log(message)
                finally:
                    if temporary is not None:
                        try:
                            temporary.unlink(missing_ok=True)
                        except OSError:
                            pass
        return copied

    def _run(self) -> None:
        while not self._stop.wait(self.every):
            try:
                self.sync_once()
            except Exception as error:  # never let the thread die silently
                message = f"[sync] unexpected {type(error).__name__}: {error}"
                self.errors.append(message)
                log(message)

    def start(self) -> "DriveSync":
        self._thread.start()
        return self

    def stop(self) -> int:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()  # wait for any in-flight copy instead of racing it
        try:
            return self.sync_once()
        except Exception as error:
            log(f"[sync] final sync failed: {type(error).__name__}: {error}")
            return 0


def checkpoint_progress(path: Path) -> dict:
    """Epoch and global step stored in a Lightning checkpoint (for logs; never fatal)."""
    try:
        import torch

        state = torch.load(str(path), map_location="cpu", weights_only=False)
        return {"epoch": state.get("epoch"), "global_step": state.get("global_step")}
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


def restore(sync_dir: Path, output: Path) -> Path:
    """Bring Drive state into ``output`` for --resume, never replacing newer local files."""
    sync_dir, output = Path(sync_dir), Path(output)
    local_last, drive_last = output / "last.ckpt", sync_dir / "last.ckpt"
    if local_last.is_file() and (not drive_last.is_file()
                                 or local_last.stat().st_mtime >= drive_last.stat().st_mtime):
        log(f"Resuming from the local {local_last} (same age or newer than Drive's copy)")
        return local_last
    if not drive_last.is_file():
        raise FileNotFoundError(f"No last.ckpt to resume from in {sync_dir} or {output}")
    output.mkdir(parents=True, exist_ok=True)
    for path in sync_dir.rglob("*"):
        if not path.is_file() or ".tmp-sync" in path.name:
            continue
        destination = output / path.relative_to(sync_dir)
        if destination.is_file() and destination.stat().st_mtime >= path.stat().st_mtime:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    log(f"Restored training state from {sync_dir}")
    return local_last


def load_model(weights: Path, device: str | None, gradient_checkpointing: bool, resolution: int | None = None):
    """Model-level settings (resolution, device, gradient checkpointing) go to the constructor."""
    from rfdetr import RFDETRLarge

    options = {"pretrain_weights": str(weights)}
    if device:
        options["device"] = device
    if gradient_checkpointing:
        options["gradient_checkpointing"] = True
    if resolution is not None:
        options["resolution"] = resolution
    return RFDETRLarge(**options)


def resolve_weights(value: str) -> Path:
    """Require a path with a folder: rfdetr maps a bare file name to ~/.roboflow/models."""
    if not os.path.dirname(str(value)):
        raise ValueError("--weights must include a folder, e.g. /content/models/rf-detr-large-2026.pth")
    return Path(value).expanduser().resolve()


def verify_base_weights(weights: Path) -> dict:
    if weights.name != LARGE_2026["file"]:
        raise ValueError(f"Base weights must be {LARGE_2026['file']} (the 2026 RF-DETR Large checkpoint)")
    actual = file_digest(weights, "md5")
    if actual != LARGE_2026["md5"]:
        raise ValueError(f"{weights} md5 {actual} != published {LARGE_2026['md5']}")
    return {"path": str(weights), "md5": actual, "sha256": file_digest(weights, "sha256")}


def checkpoint_class_names(path: Path, device: str | None) -> list[str]:
    model = load_model(path, device, False)
    names = getattr(model, "class_names", None)
    if names is None:
        raise RuntimeError("Loaded checkpoint exposes no class_names")
    return [str(name) for name in names]


def resolved_train_config(output: Path) -> dict:
    """Batch size / accumulation actually used (rfdetr replaces them when batch_size='auto')."""
    path = output / "training_config.json"
    if not path.is_file():
        return {}
    config = json.loads(path.read_text(encoding="utf-8")).get("train_config", {})
    return {key: config.get(key) for key in ("batch_size", "grad_accum_steps", "epochs", "seed",
                                             "augmentation_backend", "amp_dtype", "lr", "lr_encoder")}


def build_train_options(args, dataset: Path, output: Path, resume_path: Path | None = None) -> dict:
    """Keyword arguments for ``RFDETRLarge.train``: TrainConfig fields plus ``device``, which rfdetr's
    ``_prepare_run_config`` pops and maps to the Lightning accelerator (so ``--device`` also controls
    training, not only the constructor)."""
    options = {
        "dataset_dir": str(dataset), "output_dir": str(output), "class_names": list(EMT_CLASSES),
        "epochs": args.epochs, "batch_size": args.batch_size if args.batch_size == "auto" else int(args.batch_size),
        "grad_accum_steps": args.grad_accum, "lr": args.lr, "lr_encoder": args.lr_encoder,
        "augmentation_backend": args.augmentation_backend, "seed": args.seed,
        # > epochs keeps rfdetr's rolling last.ckpt and writes no per-epoch archives (interval 1 disables last.ckpt).
        "checkpoint_interval": max(1000, args.epochs + 1),
        "early_stopping": False, "run_test": False, "log_per_class_metrics": True,
        "num_workers": args.num_workers, "tensorboard": True, "progress_bar": "tqdm",
    }
    if args.device:
        options["device"] = args.device
    if resume_path is not None:
        options["resume"] = str(resume_path)
    return options


def train(args) -> dict:
    started = time.time()
    stamp = utc_stamp()
    output = Path(args.output)
    dataset = Path(args.dataset)
    weights = resolve_weights(args.weights)
    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available in this runtime")
    if args.sync_dir:
        require_mounted(Path(args.sync_dir))
    if args.micro:
        dataset = make_subset(dataset, Path(str(output) + "-dataset"), *args.micro)
    dataset_counts = check_dataset(dataset)
    resume_path = None
    if args.resume:
        if not args.sync_dir and not (output / "last.ckpt").is_file():
            raise ValueError("--resume needs --sync-dir or a local last.ckpt")
        resume_path = restore(Path(args.sync_dir), output) if args.sync_dir else output / "last.ckpt"
        progress = checkpoint_progress(resume_path)
        log(f"Resuming from {resume_path}: {progress}")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output {output} is not empty; use a new folder or --resume")
    output.mkdir(parents=True, exist_ok=True)

    if weights.is_file():  # rfdetr only warns on a wrong MD5, so refuse before loading
        verify_base_weights(weights)
    model = load_model(weights, args.device, args.gradient_checkpointing, args.resolution)
    base = verify_base_weights(weights)  # also covers the case where rfdetr just downloaded it
    loaded_from = getattr(getattr(model, "model_config", None), "pretrain_weights", None)
    if loaded_from is not None and Path(loaded_from).expanduser().resolve() != weights:
        raise RuntimeError(f"rfdetr loaded {loaded_from}, but {weights} was verified")
    log(f"Base weights verified: {base['path']} md5={base['md5']}")

    options = build_train_options(args, dataset, output, resume_path)
    record = {"version": 2, "mode": "micro" if args.micro else "full", "resume": bool(args.resume),
              "dataset": str(dataset), "dataset_counts": dataset_counts, "base_weights": base,
              "train_options": options,
              "model_options": {"resolution": args.resolution, "device": args.device,
                                "gradient_checkpointing": args.gradient_checkpointing},
              "code_zip": ({"path": args.code_zip, "sha256": file_digest(Path(args.code_zip), "sha256")}
                           if args.code_zip else None),
              "environment": versions(), "started_utc": stamp}
    (output / f"task1_train_request_{stamp}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    log("Training options: " + json.dumps(options))

    sync = None
    if args.sync_dir:
        sync = DriveSync(output, Path(args.sync_dir), args.sync_every_min * 60)
        if args.resume:
            sync.mark_synced()
        sync.start()
    try:
        model.train(**options)
    except BaseException as error:
        failure = dict(record, failed_utc=utc_stamp(), error=f"{type(error).__name__}: {error}",
                       last_ckpt=checkpoint_progress(output / "last.ckpt") if (output / "last.ckpt").is_file() else None)
        (output / f"task1_training_failed_{stamp}.json").write_text(json.dumps(failure, indent=2, default=str) + "\n",
                                                                     encoding="utf-8")
        log(f"Training stopped: {failure['error']}. Resume with --resume.")
        raise
    finally:
        if sync:
            sync.stop()

    best_total, best_ema = output / "checkpoint_best_total.pth", output / "checkpoint_best_ema.pth"
    selected = best_total if best_total.is_file() else best_ema
    if not selected.is_file():
        raise FileNotFoundError(f"No best checkpoint in {output}")
    names = checkpoint_class_names(selected, args.device)
    require_emt_class_order(names)
    summary = dict(record, finished_utc=utc_stamp(), finished_normally=True,
                   resolved_train_config=resolved_train_config(output),
                   selected_checkpoint={"path": str(selected), "kind": selected.name,
                                        "sha256": file_digest(selected, "sha256"), "class_names": names},
                   has_last_ckpt=(output / "last.ckpt").is_file(),
                   last_ckpt_progress=checkpoint_progress(output / "last.ckpt"),
                   drive_sync={"copies": sync.copies, "errors": sync.errors} if sync else None,
                   seconds=round(time.time() - started, 1))
    (output / "task1_training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if sync:
        sync.sync_once()  # ship the summary too
    log(f"Selected {selected.name} sha256={summary['selected_checkpoint']['sha256']}")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", required=True,
                        help=f"Path WITH a folder to {LARGE_2026['file']} (downloaded there if absent)")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", default="auto", help="'auto' or an integer")
    parser.add_argument("--grad-accum", type=int, default=None, help="Default: 16 // batch size (1 for auto)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=1.5e-4)
    parser.add_argument("--resolution", type=int, default=704)
    parser.add_argument("--augmentation-backend", default="torchvision")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--micro", nargs=2, type=int, metavar=("TRAIN", "VALID"),
                        help="Train on the first TRAIN/VALID images only (gate run)")
    parser.add_argument("--sync-dir", help="Mounted Drive folder for resumable state")
    parser.add_argument("--sync-every-min", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--code-zip", help="Code zip used for this run (its sha256 is recorded)")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.resolution % 32:
        parser.error("epochs must be >= 1 and resolution a multiple of 32 for RF-DETR Large")
    if args.grad_accum is None:
        args.grad_accum = 1 if args.batch_size == "auto" else max(1, 16 // int(args.batch_size))
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
