r"""Scene inputs for the drop-off route: one original EMT frame and the road users predicted in it.

A scene is a plain JSON-serialisable dict with four parts:

* ``model_input``: what a grounding model may see. The frame record (recording, frame, timestamp, size), the
  sha256 of the original image bytes, the predicted observations of that frame and, only if asked for,
  observations of earlier frames. Observations carry exactly the ``TrackObservation`` fields: no colour, attribute
  or confidence. Colour has to come from the image (open it with ``scene_image_path``).
* ``provenance``: the facts that tie the observations to one prediction run: track source, kind of run, kind of
  detector, vision identity and checkpoint hashes, the hashes of the run's track and provenance files, history
  length. Not for prompts.
* ``context``: descriptive details that do not change the content: the image path relative to the EMT root, profile
  name, config file, frame rate and its source, annotation access, and which manifests were checked.
* ``receipt``: how this scene was produced for this request (``origin``), when, and how long it took.

``scene_sha256`` covers ``model_input`` and ``provenance`` only, so the same content from the same run gives the
same hash on any machine and with either EMT folder layout. Track IDs are local to a recording, a track source and
one prediction run: the same frame can carry other ID numbers in another run, so anything that refers to a track ID
must also name the scene hash.

``receipt.origin`` is ``fresh_inference`` only for the scene returned by ``fresh_scene``, which ran the model for
this request; a scene loaded from a prediction folder, including a folder written by an earlier fresh run, is
``saved_predictions``. ``provenance.run_kind`` says whether that folder came from a fresh-scene run.

Saved predictions are validated with the strict loader in ``tools.vision_saved_adapters`` (run from the repository
root). Fresh inference runs the vision pipeline on every frame up to the requested one. Label files are never read;
the EMT reader reads ``annotations/metadata.txt`` for the frame rate when that file exists, and the scene says so.
Outputs of a stand-in (stub) detector are refused unless ``allow_stub=True`` (tests only). Every refusal raises
``SceneError``.

Use ``validate_scene(scene, data_root, profile=..., predictions_dir=...)`` to check a scene received from another
module: only a call with ``profile`` and ``predictions_dir`` ties the observations to a prediction run. Without them it
checks the record and the image but not where the observations came from; without ``data_root`` it checks the record
only.

    python -m trackrag.dropoff.scenes show --data-root R --recording video_054604 --frame 135 \
        --profile configs/vision/finetuned.json --predictions P --out scene.json
    python -m trackrag.dropoff.scenes export --data-root R --frames frames.json --profile ... --predictions P --out-dir D
    python -m trackrag.dropoff.scenes validate --data-root R --profile ... --predictions P D
    python -m trackrag.dropoff.scenes manifest --data-root R --recordings video_054604 video_204647 --out frames.sha256.json
"""

from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import math
import os
import re
import shutil
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

from trackrag.contracts import TrackObservation
from trackrag.emt import FrameNotFoundError, open_recording, read_release_frame_rate
from trackrag.vision.config import CONFIG_ENV

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAVED_ENV = "TRACKRAG_SAVED_PREDICTIONS"  # the same variable tools.vision_saved_adapters reads

ORIGIN_SAVED, ORIGIN_FRESH = "saved_predictions", "fresh_inference"
RUN_SAVED, RUN_FRESH = "saved_prediction_run", "fresh_scene_run"
FRESH_MARK_KEY, FRESH_MARK_VALUE = "scene_run", "fresh_scene"    # written into a fresh-scene run's provenance
DETECTOR_KINDS = ("real", "replay", "stub")
RATE_FROM_METADATA, RATE_DEFAULT = "annotations/metadata.txt", "default"
NO_ANNOTATION_ACCESS, METADATA_ACCESS = "none", "annotations/metadata.txt (frame rate only)"
BOUND_BY_MANIFEST, BOUND_BY_ID_AND_SIZE = "frame_hash_manifest", "frame_id_and_size_only"
RUN_PINNED, RUN_NOT_PINNED = "shared_run_manifest", "not_pinned"

OBSERVATION_KEYS = tuple(TrackObservation.__annotations__)
MODEL_INPUT_KEYS = ("recording_id", "frame_id", "timestamp_s", "width", "height", "image_sha256", "observations",
                    "history")
PROVENANCE_KEYS = ("track_source", "run_kind", "detector_kind", "identity_hash", "checkpoint_sha256", "tracks_sha256",
                   "provenance_sha256", "replayed_from_detections_sha256", "history_frames")
CONTEXT_KEYS = ("image_relpath", "vision_profile", "config_source", "frame_rate", "frame_rate_source",
                "annotation_access", "image_binding", "frame_manifest_sha256", "run_binding", "run_manifest_sha256")
RECEIPT_KEYS = ("origin", "created_utc", "seconds")
_RECORDING_ID = re.compile(r"[A-Za-z0-9_]+")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_IMAGE_NAME = re.compile(r"(\d+)\.(jpg|jpeg|png)", re.IGNORECASE)
_CACHE_SIZE = 8
_lock = threading.Lock()


class SceneError(ValueError):
    """A scene request or a scene record that must not be used."""


# --- small helpers -----------------------------------------------------------------------------------------------

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_bytes(path: Path) -> bytes:
    try:
        return Path(path).read_bytes()
    except OSError as error:
        raise SceneError(f"Cannot read {path}: {error}") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError as error:
        raise SceneError(f"Cannot read {path}: {error}") from error
    return digest.hexdigest()


def _json_bytes(data: bytes, what: str):
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SceneError(f"{what} is not valid JSON: {error}") from error


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True)


def _float(value) -> float:
    return float(value) + 0.0  # also turns -0.0 into 0.0, so equal numbers hash equally


def scene_sha256(scene: dict) -> str:
    """Hash of the canonical JSON of ``model_input`` and ``provenance`` (context and receipt are excluded)."""
    try:
        text = _canonical({"model_input": scene["model_input"], "provenance": scene["provenance"]})
    except (KeyError, TypeError, ValueError) as error:
        raise SceneError(f"Scene content cannot be hashed: {error}") from error
    return _sha256_bytes(text.encode("utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SceneError(message)


def _is_hex64(value) -> bool:
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def _check_relpath(value, what: str) -> None:
    _require(isinstance(value, str) and value != "" and not value.startswith("/") and "\\" not in value
             and ":" not in value and ".." not in value.split("/"), f"{what} must be a relative POSIX path, got {value!r}")


def _check_request(recording_id, frame_id, history_frames) -> None:
    _require(isinstance(recording_id, str) and _RECORDING_ID.fullmatch(recording_id) is not None,
             f"Invalid recording ID {recording_id!r}")
    _require(type(frame_id) is int and frame_id >= 0, f"Frame ID must be a nonnegative int, got {frame_id!r}")
    _require(type(history_frames) is int and history_frames >= 0,
             f"history_frames must be a nonnegative int, got {history_frames!r}")


def _resolve_profile(profile):
    """An explicit vision profile is required: a VisionConfig, a config path, or TRACKRAG_VISION_CONFIG."""
    from trackrag.vision.config import VisionConfig, load_config

    if isinstance(profile, VisionConfig):
        return profile
    chosen = profile or os.environ.get(CONFIG_ENV)
    _require(bool(chosen), f"Choose a vision profile explicitly (profile=... or {CONFIG_ENV}); "
                           "there is no silent default for scenes")
    try:
        return load_config(chosen)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise SceneError(f"Cannot load vision profile {chosen}: {error}") from error


def _config_source(config) -> str | None:
    if not config.source_path:
        return None
    path = Path(config.source_path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def _observation(row: dict) -> dict:
    """A fresh dict with exactly the contract fields and float coordinates (extra keys in saved rows are dropped)."""
    return {"recording_id": row["recording_id"], "frame_id": row["frame_id"], "track_id": row["track_id"],
            "class_name": row["class_name"], "bbox_xyxy": [_float(v) for v in row["bbox_xyxy"]],
            "source": row["source"]}


def _cache_get(cache: OrderedDict, key, build):
    with _lock:
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
    value = build()
    with _lock:
        cache[key] = value
        while len(cache) > _CACHE_SIZE:
            cache.popitem(last=False)
    return value


_artifact_cache: OrderedDict = OrderedDict()
_manifest_cache: OrderedDict = OrderedDict()
_detectors: OrderedDict = OrderedDict()


def _load_artifact(folder: Path, recording_id: str):
    """Strictly validated predictions for one recording, cached by the hashes of both files."""
    tracks_path = folder / f"{recording_id}.tracks.jsonl"
    provenance_path = folder / f"{recording_id}.provenance.json"
    _require(tracks_path.is_file() and provenance_path.is_file(), f"No saved predictions for {recording_id} in {folder}")
    provenance_bytes = _read_bytes(provenance_path)
    provenance = _json_bytes(provenance_bytes, str(provenance_path))
    _require(isinstance(provenance, dict), f"{provenance_path} must hold a JSON object")
    tracks_sha = _sha256_file(tracks_path)

    def build():
        from tools.vision_saved_adapters import load_track_artifact  # strict loader: hashes, rows, classes, boxes

        try:
            rows, frames, size, checked_sha = load_track_artifact(tracks_path, provenance, recording_id)
        except (ValueError, KeyError, TypeError, AttributeError, OSError) as error:
            raise SceneError(f"Predictions for {recording_id} in {folder} were refused: {error}") from error
        _require(checked_sha == tracks_sha, f"{tracks_path} changed while it was being read")
        by_frame: dict[int, list] = {}
        for row in rows:
            by_frame.setdefault(row["frame_id"], []).append(_observation(row))
        return by_frame, tuple(frames), tuple(size)

    key = (str(tracks_path.resolve()), tracks_sha, _sha256_bytes(provenance_bytes))
    by_frame, frames, size = _cache_get(_artifact_cache, key, build)
    return provenance, _sha256_bytes(provenance_bytes), by_frame, list(frames), list(size), tracks_sha


def _load_json_manifest(source, what: str) -> dict:
    if isinstance(source, dict):
        return copy.deepcopy(source)
    data = _read_bytes(Path(source))
    key = (str(Path(source).resolve()), _sha256_bytes(data))
    return copy.deepcopy(_cache_get(_manifest_cache, key, lambda: _json_bytes(data, f"{what} {source}")))


def load_frame_manifest(frame_manifest) -> tuple[dict, str]:
    """A validated frame-hash manifest (dict or file path) and the sha256 of its canonical JSON."""
    manifest = _load_json_manifest(frame_manifest, "Frame manifest")
    _require(isinstance(manifest, dict) and manifest.get("version") == 1 and isinstance(manifest.get("recordings"), dict),
             "A frame manifest needs version 1 and a recordings object")
    for recording_id, frames in manifest["recordings"].items():
        _require(isinstance(frames, dict) and all(name.isdigit() and _is_hex64(value) for name, value in frames.items()),
                 f"Frame manifest entries for {recording_id} must map frame numbers to sha256 values")
    return manifest, _sha256_bytes(_canonical(manifest).encode("utf-8"))


def load_run_manifest(run_manifest) -> tuple[dict, str]:
    """A validated list of known prediction-run file hashes (by identity hash and recording) and its sha256."""
    manifest = _load_json_manifest(run_manifest, "Run manifest")
    _require(isinstance(manifest, dict) and manifest.get("version") == 1 and isinstance(manifest.get("runs"), dict),
             "A run manifest needs version 1 and a runs object")
    for identity, recordings in manifest["runs"].items():
        _require(_is_hex64(identity) and isinstance(recordings, dict)
                 and all(isinstance(v, dict) and set(v) == {"tracks_sha256", "provenance_sha256"}
                         and _is_hex64(v["tracks_sha256"]) and _is_hex64(v["provenance_sha256"])
                         for v in recordings.values()),
                 "Run manifest entries need tracks_sha256 and provenance_sha256 per recording")
    return manifest, _sha256_bytes(_canonical(manifest).encode("utf-8"))


def _check_frame_manifest(frame_manifest, recording_id: str, frame_id: int, image_sha: str) -> str:
    manifest, manifest_sha = load_frame_manifest(frame_manifest)
    frames = manifest["recordings"].get(recording_id)
    _require(frames is not None and str(frame_id) in frames,
             f"The frame manifest has no entry for {recording_id} frame {frame_id}")
    _require(frames[str(frame_id)] == image_sha, f"{recording_id} frame {frame_id}: image bytes differ from the frame manifest")
    return manifest_sha


def _check_run_manifest(run_manifest, provenance: dict, recording_id: str) -> str:
    manifest, manifest_sha = load_run_manifest(run_manifest)
    entry = manifest["runs"].get(provenance["identity_hash"], {}).get(recording_id)
    _require(entry is not None, f"The run manifest has no run for {recording_id} with this vision identity")
    _require(entry == {"tracks_sha256": provenance["tracks_sha256"], "provenance_sha256": provenance["provenance_sha256"]},
             f"{recording_id}: the prediction files differ from the shared run listed in the run manifest")
    return manifest_sha


def _open(data_root, recording_id: str):
    try:
        return open_recording(data_root, recording_id)
    except (OSError, ValueError) as error:
        raise SceneError(f"Cannot open recording {recording_id} under {data_root}: {error}") from error


def _frame_image(recording, frame_id: int):
    """Frame t through the shared reader, plus the hash of its bytes. The hashed bytes are decoded with OpenCV and
    PIL, and both sizes must equal the reader's."""
    import io

    import cv2
    import numpy as np
    from PIL import Image

    _require(frame_id in recording.frame_index,
             f"Frame {frame_id} is absent from the images of recording {recording.recording_id!r}")
    path = recording.frame_index[frame_id]
    data = _read_bytes(path)
    try:
        frame = recording.get_frame(frame_id)
    except (FrameNotFoundError, ValueError) as error:
        raise SceneError(str(error)) from error
    _require(frame["recording_id"] == recording.recording_id and frame["frame_id"] == frame_id,
             "The reader returned another recording or frame")
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    _require(decoded is not None, f"Cannot decode {path.name}")
    try:
        with Image.open(io.BytesIO(data)) as image:
            pil_size = list(image.size)
    except OSError as error:
        raise SceneError(f"Cannot decode {path.name}: {error}") from error
    sizes = {"reader": [frame["width"], frame["height"]], "hashed bytes (OpenCV)": [decoded.shape[1], decoded.shape[0]],
             "hashed bytes (PIL)": pil_size}
    _require(len({tuple(v) for v in sizes.values()}) == 1,
             f"Frame {frame_id}: decoders disagree on the image size {sizes}; the image may carry an orientation tag "
             "or have changed while being read")
    relpath = (recording.image_dir.relative_to(recording.data_root) / path.name).as_posix()
    return frame, relpath, _sha256_bytes(data)


# --- building scenes ---------------------------------------------------------------------------------------------

def _build_scene(data_root, recording_id, frame_id, config, folder: Path, *, origin: str, frame_manifest,
                 run_manifest, history_frames: int, allow_stub: bool) -> dict:
    started = time.perf_counter()
    raw, provenance_sha, by_frame, predicted, size, tracks_sha = _load_artifact(folder, recording_id)
    _require(raw.get("identity_hash") == config.identity_hash(),
             f"Predictions in {folder} were made with identity {raw.get('identity_hash')}, but the chosen vision profile "
             f"{config.name!r} has identity {config.identity_hash()}")
    detector_kind = raw.get("detector_kind")
    _require(detector_kind in DETECTOR_KINDS, f"Unknown detector kind {detector_kind!r} in {folder}")
    _require(detector_kind != "stub" or allow_stub, f"Predictions in {folder} come from a stand-in (stub) detector")
    recording = _open(data_root, recording_id)
    _require(frame_id in recording.frame_index,
             f"Frame {frame_id} is absent from the images of recording {recording_id!r}")
    _require(frame_id in predicted, f"No saved predictions for {recording_id} frame {frame_id} in {folder}")
    available = [f for f in recording.frame_ids if f <= frame_id]
    _require([f for f in predicted if f <= frame_id] == available,
             f"{recording_id}: the image frames up to {frame_id} differ from the predicted frames")
    frame, relpath, image_sha = _frame_image(recording, frame_id)
    _require([frame["width"], frame["height"]] == size,
             f"{recording_id} frame {frame_id}: image size {[frame['width'], frame['height']]} differs from the "
             f"prediction size {size}")
    replayed = raw.get("replayed_from") or {}
    _require(isinstance(replayed, dict), f"{recording_id}: replayed_from in the provenance file must be an object")
    provenance = {
        "track_source": "predicted",
        "run_kind": RUN_FRESH if raw.get(FRESH_MARK_KEY) == FRESH_MARK_VALUE else RUN_SAVED,
        "detector_kind": detector_kind, "identity_hash": config.identity_hash(),
        "checkpoint_sha256": config.detector.checkpoint_sha256, "tracks_sha256": tracks_sha,
        "provenance_sha256": provenance_sha, "replayed_from_detections_sha256": replayed.get("detections_sha256"),
        "history_frames": history_frames,
    }
    frame_manifest_sha = (_check_frame_manifest(frame_manifest, recording_id, frame_id, image_sha)
                          if frame_manifest is not None else None)
    run_manifest_sha = (_check_run_manifest(run_manifest, provenance, recording_id)
                        if run_manifest is not None else None)
    earlier = [f for f in predicted if f < frame_id][-history_frames:] if history_frames else []
    history = [_observation(o) for f in earlier for o in sorted(by_frame.get(f, []), key=lambda o: o["track_id"])]
    observations = [_observation(o) for o in sorted(by_frame.get(frame_id, []), key=lambda o: o["track_id"])]
    declared_rate = read_release_frame_rate(recording.data_root)
    metadata = Path(recording.data_root) / "annotations" / "metadata.txt"
    scene = {
        "model_input": {
            "recording_id": recording_id, "frame_id": frame_id, "timestamp_s": _float(frame["timestamp_s"]),
            "width": frame["width"], "height": frame["height"], "image_sha256": image_sha,
            "observations": observations, "history": history,
        },
        "provenance": provenance,
        "context": {
            "image_relpath": relpath, "vision_profile": config.name, "config_source": _config_source(config),
            "frame_rate": _float(recording.frame_rate),
            "frame_rate_source": RATE_FROM_METADATA if declared_rate else RATE_DEFAULT,
            "annotation_access": METADATA_ACCESS if metadata.is_file() else NO_ANNOTATION_ACCESS,
            "image_binding": BOUND_BY_MANIFEST if frame_manifest_sha else BOUND_BY_ID_AND_SIZE,
            "frame_manifest_sha256": frame_manifest_sha,
            "run_binding": RUN_PINNED if run_manifest_sha else RUN_NOT_PINNED,
            "run_manifest_sha256": run_manifest_sha,
        },
        "receipt": {"origin": origin, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "seconds": round(time.perf_counter() - started, 3)},
    }
    scene["scene_sha256"] = scene_sha256(scene)
    return scene


def load_scene(data_root, recording_id: str, frame_id: int, *, profile=None, predictions_dir=None,
               frame_manifest=None, run_manifest=None, history_frames: int = 0, allow_stub: bool = False) -> dict:
    """The scene for one frame from a prediction folder made with the chosen vision profile (origin
    ``saved_predictions``)."""
    _check_request(recording_id, frame_id, history_frames)
    config = _resolve_profile(profile)
    folder = predictions_dir or os.environ.get(SAVED_ENV)
    _require(bool(folder), f"Give the saved prediction folder (predictions_dir=... or {SAVED_ENV})")
    return _build_scene(data_root, recording_id, frame_id, config, Path(folder), origin=ORIGIN_SAVED,
                        frame_manifest=frame_manifest, run_manifest=run_manifest, history_frames=history_frames,
                        allow_stub=allow_stub)


def fresh_scene(data_root, recording_id: str, frame_id: int, *, profile=None, out_dir, weights_dir=None,
                device=None, detector=None, tracker_factory=None, frame_manifest=None, history_frames: int = 0,
                allow_stub: bool = False) -> dict:
    """Run the vision pipeline on every frame up to ``frame_id`` for this request (origin ``fresh_inference``),
    save that run in ``out_dir`` (a new folder) and build the scene from it through the same strict path as saved
    predictions. ``detector`` and ``tracker_factory`` replace the real models in tests."""
    from trackrag.vision import predict_tracks
    from trackrag.vision.__main__ import JsonlSink, PrefixRecording

    _check_request(recording_id, frame_id, history_frames)
    config = _resolve_profile(profile).with_overrides(weights_dir=weights_dir, device=device)
    out = Path(out_dir)
    partial = out.with_name(out.name + ".partial")
    _require(not out.exists(), f"Output folder already exists: {out} (choose a new one)")
    _require(not partial.exists(), f"A previous fresh run left {partial}; delete it, then run again")
    recording = _open(data_root, recording_id)
    _require(frame_id in recording.frame_index,
             f"Frame {frame_id} is absent from the images of recording {recording_id!r}")
    _, _, image_sha = _frame_image(recording, frame_id)  # refuse a wrong image before running the model
    if frame_manifest is not None:
        _check_frame_manifest(frame_manifest, recording_id, frame_id, image_sha)
    prefix = PrefixRecording(recording, recording.frame_ids.index(frame_id) + 1)
    if detector is None:
        def build():
            from trackrag.vision.detector import RFDETRDetector

            return RFDETRDetector(config)

        detector = _cache_get(_detectors, (config.identity_hash(), str(config.weights_dir), str(config.device)), build)
        with _lock:
            while len(_detectors) > 2:
                _detectors.popitem(last=False)
    partial.mkdir(parents=True)
    sink = JsonlSink(partial, recording_id)
    try:
        predict_tracks(prefix, config=config, detector=detector, tracker_factory=tracker_factory, sink=sink)
        provenance_path = partial / f"{recording_id}.provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance[FRESH_MARK_KEY] = FRESH_MARK_VALUE
        provenance_path.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8",
                                   newline="\n")
    except BaseException:
        sink.close()
        shutil.rmtree(partial, ignore_errors=True)
        raise
    partial.rename(out)
    return _build_scene(data_root, recording_id, frame_id, config, out, origin=ORIGIN_FRESH,
                        frame_manifest=frame_manifest, run_manifest=None, history_frames=history_frames,
                        allow_stub=allow_stub)


def export_scenes(data_root, frames, out_dir, *, profile=None, predictions_dir=None, frame_manifest=None,
                  run_manifest=None, history_frames: int = 0, allow_stub: bool = False) -> list[dict]:
    """One scene file per (recording_id, frame_id) pair, named <recording>_f<frame>.json, plus index.json."""
    out = Path(out_dir)
    _require(not out.exists(), f"Output folder already exists: {out} (choose a new one)")
    _require(isinstance(frames, list) and len(frames) > 0, "The frame list must be a nonempty JSON list")
    for item in frames:
        _require(isinstance(item, dict) and set(item) == {"recording_id", "frame_id"},
                 f'Each frame entry needs exactly "recording_id" and "frame_id": {item!r}')
        _check_request(item["recording_id"], item["frame_id"], history_frames)
    pairs = [(item["recording_id"], item["frame_id"]) for item in frames]
    _require(len(set(pairs)) == len(pairs), "The frame list repeats a (recording, frame) pair")
    built = [load_scene(data_root, recording_id, frame_id, profile=profile, predictions_dir=predictions_dir,
                        frame_manifest=frame_manifest, run_manifest=run_manifest, history_frames=history_frames,
                        allow_stub=allow_stub) for recording_id, frame_id in pairs]
    out.mkdir(parents=True)
    index = []
    for scene in built:
        name = f"{scene['model_input']['recording_id']}_f{scene['model_input']['frame_id']}.json"
        (out / name).write_text(json.dumps(scene, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        index.append({"file": name, "scene_sha256": scene["scene_sha256"]})
    (out / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8", newline="\n")
    return built


def scene_image_path(scene: dict, data_root) -> Path:
    """The original image of the scene's recording and frame under ``data_root`` (found through the EMT reader),
    after checking the scene record and the image bytes."""
    _check_structure(scene)
    model_input = scene["model_input"]
    recording = _open(data_root, model_input["recording_id"])
    _require(model_input["frame_id"] in recording.frame_index,
             f"Frame {model_input['frame_id']} is absent from the images of recording {model_input['recording_id']!r}")
    path = recording.frame_index[model_input["frame_id"]]
    _require(_sha256_file(path) == model_input["image_sha256"], f"{path} does not match the scene's image hash")
    return path


def frame_hash_manifest(data_root, recording_ids) -> dict:
    """sha256 of every frame image of the given recordings in the given EMT root."""
    manifest = {"version": 1, "recordings": {}}
    for recording_id in recording_ids:
        _check_request(recording_id, 0, 0)
        recording = _open(data_root, recording_id)
        manifest["recordings"][recording_id] = {
            str(frame_id): _sha256_file(recording.frame_index[frame_id]) for frame_id in recording.frame_ids}
    return manifest


# --- validation at a module boundary -----------------------------------------------------------------------------

def _check_box(box, width: int, height: int) -> None:
    _require(isinstance(box, list) and len(box) == 4 and all(type(v) is float and math.isfinite(v) for v in box),
             f"A box needs four finite float coordinates, got {box!r}")
    left, top, right, bottom = box
    _require(0 <= left < right <= width and 0 <= top < bottom <= height,
             f"Box {box} lies outside the {width}x{height} image or has no area")


def _check_structure(scene) -> None:
    from trackrag.vision.classes import EMT_CLASSES

    _require(isinstance(scene, dict) and set(scene) == {"scene_sha256", "model_input", "provenance", "context", "receipt"},
             "A scene has exactly scene_sha256, model_input, provenance, context and receipt")
    model_input, provenance, context, receipt = (scene[k] for k in ("model_input", "provenance", "context", "receipt"))
    _require(isinstance(model_input, dict) and set(model_input) == set(MODEL_INPUT_KEYS), "Unexpected model_input fields")
    _require(isinstance(provenance, dict) and set(provenance) == set(PROVENANCE_KEYS), "Unexpected provenance fields")
    _require(isinstance(context, dict) and set(context) == set(CONTEXT_KEYS), "Unexpected context fields")
    _require(isinstance(receipt, dict) and set(receipt) == set(RECEIPT_KEYS), "Unexpected receipt fields")
    recording_id, frame_id = model_input["recording_id"], model_input["frame_id"]
    _check_request(recording_id, frame_id, provenance["history_frames"])
    width, height, timestamp = model_input["width"], model_input["height"], model_input["timestamp_s"]
    _require(type(width) is int and type(height) is int and width > 0 and height > 0, "Invalid image size")
    _require(type(timestamp) is float and math.isfinite(timestamp) and timestamp >= 0,
             f"timestamp_s must be a finite nonnegative float, got {timestamp!r}")
    _require(_is_hex64(model_input["image_sha256"]), "Invalid image hash")
    _require(provenance["track_source"] == "predicted", "Scenes carry predicted observations only")
    _require(provenance["run_kind"] in (RUN_SAVED, RUN_FRESH), f"Unknown run kind {provenance['run_kind']!r}")
    _require(provenance["detector_kind"] in DETECTOR_KINDS, f"Unknown detector kind {provenance['detector_kind']!r}")
    for name in ("identity_hash", "tracks_sha256", "provenance_sha256"):
        _require(_is_hex64(provenance[name]), f"provenance.{name} must be a sha256 value")
    for name in ("checkpoint_sha256", "replayed_from_detections_sha256"):
        _require(provenance[name] is None or _is_hex64(provenance[name]), f"provenance.{name} must be a sha256 or null")
    relpath = context["image_relpath"]
    _check_relpath(relpath, "context.image_relpath")
    parts = relpath.split("/")
    name_match = _IMAGE_NAME.fullmatch(parts[-1])
    _require(len(parts) >= 2 and parts[-2] == recording_id and name_match is not None
             and int(name_match.group(1)) == frame_id,
             f"context.image_relpath {relpath!r} does not name frame {frame_id} of {recording_id}")
    _require(isinstance(context["vision_profile"], str) and context["vision_profile"] != "", "context.vision_profile missing")
    if context["config_source"] is not None:
        _check_relpath(context["config_source"], "context.config_source")
    rate = context["frame_rate"]
    _require(type(rate) is float and math.isfinite(rate) and rate > 0, "context.frame_rate must be a positive float")
    _require(context["frame_rate_source"] in (RATE_FROM_METADATA, RATE_DEFAULT), "Unknown frame_rate_source")
    _require(context["annotation_access"] in (NO_ANNOTATION_ACCESS, METADATA_ACCESS), "Unknown annotation_access")
    for flag, sha, bound, unbound in (("image_binding", "frame_manifest_sha256", BOUND_BY_MANIFEST, BOUND_BY_ID_AND_SIZE),
                                      ("run_binding", "run_manifest_sha256", RUN_PINNED, RUN_NOT_PINNED)):
        _require((context[flag] == bound and _is_hex64(context[sha])) or (context[flag] == unbound and context[sha] is None),
                 f"context.{flag} and context.{sha} disagree")
    _require(receipt["origin"] in (ORIGIN_SAVED, ORIGIN_FRESH), f"Unknown origin {receipt['origin']!r}")
    _require(receipt["origin"] == ORIGIN_SAVED or provenance["run_kind"] == RUN_FRESH,
             "A fresh_inference scene must come from a fresh-scene run")
    seconds = receipt["seconds"]
    _require(isinstance(receipt["created_utc"], str)
             and (type(seconds) is int or (type(seconds) is float and math.isfinite(seconds))) and seconds >= 0,
             "Invalid receipt")
    seen = set()
    for name, rows in (("observations", model_input["observations"]), ("history", model_input["history"])):
        _require(isinstance(rows, list), f"{name} must be a list")
        for row in rows:
            _require(isinstance(row, dict) and set(row) == set(OBSERVATION_KEYS),
                     "An observation has exactly the six TrackObservation fields")
            _require(row["recording_id"] == recording_id, "An observation belongs to another recording")
            _require(row["source"] == "predicted", "An observation comes from another track source")
            _require(type(row["frame_id"]) is int and type(row["track_id"]) is int and row["track_id"] >= 0
                     and row["frame_id"] >= 0, "Observation frame and track IDs must be nonnegative ints")
            if name == "observations":
                _require(row["frame_id"] == frame_id,
                         f"Observation from frame {row['frame_id']} in a scene for frame {frame_id}")
            else:
                _require(row["frame_id"] < frame_id,
                         f"History observation from frame {row['frame_id']} is not earlier than frame {frame_id}")
            _require(row["class_name"] in EMT_CLASSES, f"Unknown class {row['class_name']!r}")
            _check_box(row["bbox_xyxy"], width, height)
            key = (row["frame_id"], row["track_id"])
            _require(key not in seen, f"Duplicate observation {key}")
            seen.add(key)
    _require(len({row["frame_id"] for row in model_input["history"]}) <= provenance["history_frames"],
             "History covers more frames than declared")
    _require(scene["scene_sha256"] == scene_sha256(scene), "scene_sha256 does not match the scene content")


def validate_scene(scene: dict, data_root=None, *, profile=None, predictions_dir=None, frame_manifest=None,
                   run_manifest=None, allow_stub: bool = False) -> None:
    """Check a scene record (also after a JSON round trip); see the module docstring for what each set of arguments proves.

    Always: fields, types, values, the image path naming this frame, observations at the scene frame only, history
    before it, stub refusal, the hash. ``frame_manifest`` / ``run_manifest``: the image hash and the run files against
    those lists. ``data_root``: the frame and image re-read through the EMT reader. ``profile`` and
    ``predictions_dir`` (with ``data_root``): the scene must be exactly what that prediction folder gives."""
    _check_structure(scene)
    model_input, provenance, context = scene["model_input"], scene["provenance"], scene["context"]
    recording_id, frame_id = model_input["recording_id"], model_input["frame_id"]
    _require(provenance["detector_kind"] != "stub" or allow_stub, "The scene comes from a stand-in (stub) detector")
    if frame_manifest is not None:
        manifest_sha = _check_frame_manifest(frame_manifest, recording_id, frame_id, model_input["image_sha256"])
        _require(context["frame_manifest_sha256"] in (None, manifest_sha),
                 "context.frame_manifest_sha256 names another frame manifest")
    if run_manifest is not None:
        manifest_sha = _check_run_manifest(run_manifest, provenance, recording_id)
        _require(context["run_manifest_sha256"] in (None, manifest_sha), "context.run_manifest_sha256 names another run manifest")
    if data_root is None:
        _require(profile is None and predictions_dir is None, "Comparing with predictions needs data_root")
        return
    recording = _open(data_root, recording_id)
    frame, _, image_sha = _frame_image(recording, frame_id)
    _require(image_sha == model_input["image_sha256"], "Image bytes differ from the scene")
    _require([frame["width"], frame["height"]] == [model_input["width"], model_input["height"]],
             "Frame size differs from the reader")
    _require(_float(frame["timestamp_s"]) == model_input["timestamp_s"],
             f"Timestamp differs from the reader ({frame['timestamp_s']} at {recording.frame_rate} fps)")
    _require({row["frame_id"] for row in model_input["history"]} <= set(recording.frame_ids),
             "History refers to frames absent from the recording")
    if profile is None and predictions_dir is None:
        return
    expected = load_scene(data_root, recording_id, frame_id, profile=profile, predictions_dir=predictions_dir,
                          frame_manifest=frame_manifest, run_manifest=run_manifest,
                          history_frames=provenance["history_frames"], allow_stub=allow_stub)
    differing = [f"model_input.{k}" for k in MODEL_INPUT_KEYS if expected["model_input"][k] != model_input[k]]
    differing += [f"provenance.{k}" for k in PROVENANCE_KEYS if expected["provenance"][k] != provenance[k]]
    differing += [f"context.{k}" for k in ("frame_rate", "frame_rate_source", "annotation_access")
                  if expected["context"][k] != context[k]]
    _require(not differing, f"The scene differs from what the given predictions produce in {differing}")
    _require(expected["scene_sha256"] == scene["scene_sha256"], "scene_sha256 differs from the predictions' scene")


# --- command line ------------------------------------------------------------------------------------------------

def _scene_files(paths) -> list[Path]:
    """Scene files from files, folders (checked against their index.json) or wildcard patterns."""
    files = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            found = sorted(p for p in path.glob("*.json") if p.name != "index.json")
            index_path = path / "index.json"
            if index_path.is_file():
                index = _json_bytes(_read_bytes(index_path), str(index_path))
                listed = [e.get("file") for e in index if isinstance(e, dict)] if isinstance(index, list) else []
                _require(isinstance(index, list) and len(listed) == len(index) and all(isinstance(f, str) for f in listed)
                         and sorted(listed) == [p.name for p in found], f"{path}: the scene files differ from index.json")
            files += found
        elif any(ch in item for ch in "*?["):
            files += sorted(Path(p) for p in glob.glob(item))
        else:
            files.append(path)
    _require(len(files) > 0, "No scene files given")
    return files


def _check_new_file(out) -> None:
    if out and Path(out).exists():
        raise SceneError(f"{out} already exists (choose a new name)")


def _write_json(value, out) -> None:
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if out:
        Path(out).write_text(text, encoding="utf-8", newline="\n")
    else:
        sys.stdout.write(text)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m trackrag.dropoff.scenes",
                                     description="Scene inputs for the drop-off route (see the module docstring).")
    commands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--profile", help=f"Vision config JSON (or {CONFIG_ENV})")
    common.add_argument("--predictions", help=f"Prediction folder (or {SAVED_ENV})")
    common.add_argument("--frame-manifest", help="Frame image hashes made with the manifest command")
    common.add_argument("--run-manifest", help="Known prediction-run file hashes")
    common.add_argument("--allow-stub", action="store_true", help="Accept stand-in detector outputs (tests only)")
    show = commands.add_parser("show", parents=[common], help="Write the scene for one frame")
    show.add_argument("--data-root", required=True, help="EMT root (images only is enough)")
    show.add_argument("--recording", required=True)
    show.add_argument("--frame", type=int, required=True)
    show.add_argument("--history", type=int, default=0, help="Also include the last N earlier predicted frames")
    show.add_argument("--out", help="Output JSON file (default: print)")
    export = commands.add_parser("export", parents=[common], help="Write scenes for a list of frames")
    export.add_argument("--data-root", required=True, help="EMT root (images only is enough)")
    export.add_argument("--frames", required=True, help='JSON list of {"recording_id": ..., "frame_id": ...}')
    export.add_argument("--history", type=int, default=0, help="Also include the last N earlier predicted frames")
    export.add_argument("--out-dir", required=True)
    manifest = commands.add_parser("manifest", help="Hash every frame image of some recordings")
    manifest.add_argument("--data-root", required=True)
    manifest.add_argument("--recordings", nargs="+", required=True)
    manifest.add_argument("--out", required=True)
    validate = commands.add_parser("validate", parents=[common],
                                   help="Check scene files, folders of scene files or wildcard patterns")
    validate.add_argument("scenes", nargs="+")
    validate.add_argument("--data-root", help="Re-read each frame and image (needed for --profile/--predictions)")
    args = parser.parse_args(argv)
    try:
        if args.command == "show":
            _check_new_file(args.out)
            scene = load_scene(args.data_root, args.recording, args.frame, profile=args.profile,
                               predictions_dir=args.predictions, frame_manifest=args.frame_manifest,
                               run_manifest=args.run_manifest, history_frames=args.history, allow_stub=args.allow_stub)
            _write_json(scene, args.out)
        elif args.command == "export":
            frames = _json_bytes(_read_bytes(Path(args.frames)), f"Frame list {args.frames}")
            written = export_scenes(args.data_root, frames, args.out_dir, profile=args.profile,
                                    predictions_dir=args.predictions, frame_manifest=args.frame_manifest,
                                    run_manifest=args.run_manifest, history_frames=args.history,
                                    allow_stub=args.allow_stub)
            print(f"Wrote {len(written)} scenes to {args.out_dir}")
        elif args.command == "manifest":
            _check_new_file(args.out)
            _write_json(frame_hash_manifest(args.data_root, args.recordings), args.out)
        else:
            frame_manifest = load_frame_manifest(args.frame_manifest)[0] if args.frame_manifest else None
            run_manifest = load_run_manifest(args.run_manifest)[0] if args.run_manifest else None
            for path in _scene_files(args.scenes):
                scene = _json_bytes(_read_bytes(path), str(path))
                validate_scene(scene, args.data_root, profile=args.profile, predictions_dir=args.predictions,
                               frame_manifest=frame_manifest, run_manifest=run_manifest, allow_stub=args.allow_stub)
                print(f"ok: {path}")
    except SceneError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
