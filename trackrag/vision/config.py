"""Task 1 vision configuration: which checkpoint, which preprocessing, which tracker settings.

A config file names the model and its settings; *where* the weights live on a machine is
not part of it (``TRACKRAG_VISION_WEIGHTS_DIR`` or ``--weights-dir``), so the same config
and the same identity hash are valid on Colab and on a laptop.

The identity hash covers everything that changes predictions: checkpoint hash, label space,
preprocessing, thresholds and tracker settings. Saved predictions record it, and the saved
prediction adapter refuses files produced under a different identity.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

PIPELINE_VERSION = 2  # 2: camera-car (ego) filter
CONTINUITY_KIND = "ultralytics_botsort_continuity_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "vision" / "pretrained.json"
CONFIG_ENV = "TRACKRAG_VISION_CONFIG"
WEIGHTS_ENV = "TRACKRAG_VISION_WEIGHTS_DIR"
DEVICE_ENV = "TRACKRAG_VISION_DEVICE"
PLACEHOLDER = "FILL_IN_AFTER_TRAINING"
LABEL_SPACES = ("coco", "emt")
_HEX32 = re.compile(r"^[0-9a-f]{32}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class DetectorSettings:
    label_space: str
    checkpoint: str
    checkpoint_md5: str | None = None
    checkpoint_sha256: str | None = None
    kind: str = "rfdetr_large"
    threshold: float = 0.05
    resolution: int = 704
    inference_long_side: int = 1280
    antialias: bool = True
    nms_iou: float = 0.8
    rider_merge_iou: float = 0.1
    # Camera-car (ego vehicle) filter: drop a box whose bottom edge is in the last 3% of the frame and
    # whose width is >= 60% of the frame. EMT labels never contain the ego vehicle (Task 2 skips AV
    # rows); this rule removes 0 of all 576,374 real EMT label boxes (widest real one: 40.6%).
    ego_bottom_fraction: float = 0.97
    ego_min_width_fraction: float = 0.6

    def validate(self) -> None:
        if not 0.5 <= self.ego_bottom_fraction <= 1.0 or not 0.3 <= self.ego_min_width_fraction <= 1.0:
            raise ValueError("ego_bottom_fraction must be in [0.5, 1] and ego_min_width_fraction in [0.3, 1]")
        if self.kind != "rfdetr_large":
            raise ValueError(f"Unsupported detector kind {self.kind!r}")
        if self.label_space not in LABEL_SPACES:
            raise ValueError(f"label_space must be one of {LABEL_SPACES}")
        if not self.checkpoint or Path(self.checkpoint).name != self.checkpoint:
            raise ValueError("checkpoint must be a bare file name; its folder comes from the weights directory")
        for name in ("checkpoint_md5", "checkpoint_sha256"):
            value = getattr(self, name)
            if value is not None and value == PLACEHOLDER:
                raise ValueError(f"{name} is still the placeholder {PLACEHOLDER!r}: fill in the trained checkpoint's hash")
        if self.checkpoint_md5 is not None and not _HEX32.match(self.checkpoint_md5):
            raise ValueError("checkpoint_md5 must be 32 lowercase hex characters")
        if self.checkpoint_sha256 is not None and not _HEX64.match(self.checkpoint_sha256):
            raise ValueError("checkpoint_sha256 must be 64 lowercase hex characters")
        if self.label_space == "emt" and self.checkpoint_sha256 is None:
            raise ValueError("A fine-tuned (emt) checkpoint must be pinned by checkpoint_sha256")
        if self.checkpoint_md5 is None and self.checkpoint_sha256 is None:
            raise ValueError("Pin the checkpoint by checkpoint_md5 or checkpoint_sha256")
        if not 0.0 <= self.threshold < 1.0 or not 0.0 < self.nms_iou <= 1.0 or not 0.0 <= self.rider_merge_iou < 1.0:
            raise ValueError("Invalid detector threshold, NMS IoU or rider-merge IoU")
        if self.resolution % 32 or self.resolution < 32 or self.inference_long_side < 32:
            raise ValueError("resolution must be a positive multiple of 32 and inference_long_side >= 32")


@dataclass(frozen=True)
class TrackerSettings:
    kind: str = "ultralytics_botsort"
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    fuse_score: bool = True
    gmc_method: str = "sparseOptFlow"
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.8
    with_reid: bool = False

    def validate(self) -> None:
        if self.kind not in ("ultralytics_botsort", CONTINUITY_KIND):
            raise ValueError(f"Unsupported tracker kind {self.kind!r}")
        if self.with_reid:
            raise ValueError("ReID is not used in Task 1")
        if not 0 <= self.track_low_thresh <= self.track_high_thresh <= 1 or self.track_buffer < 1:
            raise ValueError("Invalid tracker thresholds or buffer")
        for name in ("new_track_thresh", "match_thresh", "proximity_thresh", "appearance_thresh"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"tracker {name} must be within [0, 1]")
        if type(self.fuse_score) is not bool:
            raise ValueError("tracker fuse_score must be true or false")

    def as_namespace_dict(self) -> dict:
        values = asdict(self)
        values.pop("kind")
        values["tracker_type"] = "botsort"
        values["model"] = "auto"
        return values


@dataclass(frozen=True)
class VisionConfig:
    name: str
    detector: DetectorSettings
    tracker: TrackerSettings = field(default_factory=TrackerSettings)
    weights_dir: str | None = None   # machine-specific: not part of the identity
    device: str | None = None        # machine-specific: not part of the identity
    source_path: str | None = None

    def identity(self) -> dict:
        return {"pipeline_version": PIPELINE_VERSION, "detector": asdict(self.detector), "tracker": asdict(self.tracker)}

    def identity_hash(self) -> str:
        canonical = json.dumps(self.identity(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def checkpoint_path(self) -> Path:
        directory = self.weights_dir or os.environ.get(WEIGHTS_ENV)
        if not directory:
            raise ValueError(f"Set the weights folder with --weights-dir or {WEIGHTS_ENV}")
        return Path(directory) / self.detector.checkpoint

    def with_overrides(self, *, weights_dir=None, device=None, fuse_score=None, tracker=None) -> "VisionConfig":
        """Return a copy with machine settings or tracker settings changed (tracker changes alter the identity)."""
        config = self
        if weights_dir is not None:
            config = replace(config, weights_dir=str(weights_dir))
        if device is not None:
            config = replace(config, device=str(device))
        if fuse_score is not None:
            config = replace(config, tracker=replace(config.tracker, fuse_score=bool(fuse_score)))
        if tracker:
            known = {f.name for f in fields(TrackerSettings)}
            unknown = sorted(set(tracker) - known)
            if unknown:
                raise ValueError(f"Unknown tracker settings: {unknown}")
            config = replace(config, tracker=replace(config.tracker, **_normalise_types(tracker, TrackerSettings, "tracker")))
        config.tracker.validate()
        return config


def _normalise_types(values: dict, cls, where: str) -> dict:
    """Reject wrong JSON types (e.g. "false" for a bool) and store numbers as float/int consistently,
    so 1 and 1.0 give the same identity hash."""
    expected = {f.name: str(f.type) for f in fields(cls)}
    normalised = {}
    for name, value in values.items():
        kind = expected[name]
        optional = "None" in kind
        if value is None and optional:
            normalised[name] = None
        elif kind.startswith("bool"):
            if type(value) is not bool:
                raise ValueError(f"{where}.{name} must be true or false, got {value!r}")
            normalised[name] = value
        elif kind.startswith("float"):
            if type(value) not in (int, float):
                raise ValueError(f"{where}.{name} must be a number, got {value!r}")
            normalised[name] = float(value)
        elif kind.startswith("int"):
            if type(value) is not int:
                raise ValueError(f"{where}.{name} must be an integer, got {value!r}")
            normalised[name] = value
        elif kind.startswith("str"):
            if type(value) is not str:
                raise ValueError(f"{where}.{name} must be text, got {value!r}")
            normalised[name] = value
        else:
            normalised[name] = value
    return normalised


def _build(cls, values: dict, where: str):
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"Unknown {where} settings: {unknown}")
    return cls(**_normalise_types(values, cls, where))


def config_from_dict(data: dict, *, source_path: str | None = None) -> VisionConfig:
    unknown = sorted(set(data) - {"name", "detector", "tracker", "notes"})
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    detector = _build(DetectorSettings, dict(data["detector"]), "detector")
    tracker = _build(TrackerSettings, dict(data.get("tracker", {})), "tracker")
    detector.validate()
    tracker.validate()
    return VisionConfig(name=str(data["name"]), detector=detector, tracker=tracker, source_path=source_path)


def load_config(path=None, *, weights_dir=None, device=None) -> VisionConfig:
    """Load a config file (default: env ``TRACKRAG_VISION_CONFIG`` or configs/vision/pretrained.json)."""
    chosen = Path(path or os.environ.get(CONFIG_ENV) or DEFAULT_CONFIG)
    data = json.loads(chosen.read_text(encoding="utf-8"))
    config = config_from_dict(data, source_path=str(chosen))
    return config.with_overrides(weights_dir=weights_dir, device=device or os.environ.get(DEVICE_ENV) or None)
