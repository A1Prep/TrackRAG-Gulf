"""RF-DETR Large adapter. Model libraries are imported only when a detector is built."""

from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path

from .classes import require_emt_class_order
from .config import VisionConfig
from .detections import Detection


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint(config: VisionConfig) -> dict:
    """Refuse a missing or different checkpoint *before* any model library sees it.

    rfdetr only warns on a wrong MD5 and silently downloads a missing registered file, so the
    pinned hashes in the config are checked here first.
    """
    path = config.checkpoint_path()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path} (set --weights-dir / TRACKRAG_VISION_WEIGHTS_DIR)")
    record = {"path": str(path), "file": path.name}
    if config.detector.checkpoint_md5:
        actual = file_digest(path, "md5")
        if actual != config.detector.checkpoint_md5:
            raise ValueError(f"{path} md5 {actual} != pinned {config.detector.checkpoint_md5}")
        record["md5"] = actual
    if config.detector.checkpoint_sha256:
        actual = file_digest(path, "sha256")
        if actual != config.detector.checkpoint_sha256:
            raise ValueError(f"{path} sha256 {actual} != pinned {config.detector.checkpoint_sha256}")
        record["sha256"] = actual
    return record


def resolve_device(requested: str | None) -> str:
    import torch

    if requested is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Device 'cuda' requested but CUDA is not available in this runtime")
    if requested not in ("cpu",) and not requested.startswith("cuda"):
        raise ValueError(f"Unsupported device {requested!r}; use 'cuda' or 'cpu'")
    return requested


class RFDETRDetector:
    """Runs RF-DETR on an RGB PIL image and returns detections in that image's pixels."""

    kind = "real"

    def __init__(self, config: VisionConfig):
        self.settings = config.detector
        self.checkpoint = verify_checkpoint(config)
        import torch
        from rfdetr import RFDETRLarge

        self.device = resolve_device(config.device)
        self.model = RFDETRLarge(pretrain_weights=self.checkpoint["path"], device=self.device,
                                 resolution=self.settings.resolution)
        self.class_names = [str(name) for name in self.model.class_names]
        if self.settings.label_space == "emt":
            require_emt_class_order(self.class_names)
        self._torch = torch
        self.gpu = torch.cuda.get_device_name(0) if self.device.startswith("cuda") else None

    def describe(self) -> dict:
        versions = {}
        for package in ("torch", "torchvision", "rfdetr", "ultralytics", "supervision", "numpy", "pillow", "lap"):
            try:
                versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                versions[package] = None
        return {"kind": self.kind, "checkpoint": self.checkpoint, "device": self.device, "gpu": self.gpu,
                "label_space": self.settings.label_space, "class_names": self.class_names, "packages": versions}

    def detect(self, image) -> list[Detection]:
        width, height = image.size
        with self._torch.inference_mode():
            result = self.model.predict(image, threshold=self.settings.threshold,
                                        include_source_image=False, antialias=self.settings.antialias)
        if len(result) == 0:
            return []
        shapes = result.data.get("source_shape")
        if shapes is None or any(tuple(int(v) for v in row) != (height, width) for row in shapes):
            raise RuntimeError(f"RF-DETR source_shape does not match the {width}x{height} input image")
        names = result.data.get("class_name")
        if names is None:
            raise RuntimeError("RF-DETR output carries no class names")
        detections = []
        for box, score, name in zip(result.xyxy, result.confidence, names):
            left, top, right, bottom = (float(value) for value in box)
            detections.append(Detection((left, top, max(left, right), max(top, bottom)), float(score), str(name)))
        return detections
