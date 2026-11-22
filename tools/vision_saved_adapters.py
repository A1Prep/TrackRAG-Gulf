"""Explicit ``--adapters`` module: the real Task 2/3/4 functions plus Task 1 predictions loaded from files.

For teammates without a GPU, and for demos, after predictions were made on Colab with
``python -m trackrag.vision predict``:

    TRACKRAG_SAVED_PREDICTIONS=<prediction folder> \
    python -m trackrag review --adapters tools.vision_saved_adapters --source predicted ...

``run.json`` records this adapter module, so saved-prediction runs stay distinguishable from
live ``trackrag.vision`` runs. Loading is refused unless the folder's provenance identity hash
matches the active vision config (``TRACKRAG_VISION_CONFIG``, default pretrained) and the
recording's frames are a prefix of the predicted frames. Prefixes are valid because Task 1's
output is online: rows for frame t never depend on later frames.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path

from trackrag.answering import answer_question  # noqa: F401  (re-exported real Task 3 answerer)
from trackrag.emt import open_recording  # noqa: F401  (re-exported real Task 2 reader)
from trackrag.questions import load_questions  # noqa: F401  (re-exported real Task 3 loader)
from trackrag.retrieval import build_index, rank_evidence  # noqa: F401  (re-exported real Task 4)

SAVED_ENV = "TRACKRAG_SAVED_PREDICTIONS"


def validated_frame_ids(values) -> list[int]:
    if (not isinstance(values, list) or not values
            or any(type(value) is not int or value < 0 for value in values)
            or values != sorted(set(values))):
        raise ValueError("Frame IDs must be nonempty, unique, increasing nonnegative integers")
    return values


def validated_box(values, width, height) -> tuple[float, float, float, float]:
    if (not isinstance(values, (list, tuple)) or len(values) != 4
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
        raise ValueError("Prediction boxes require four finite numeric coordinates")
    left, top, right, bottom = values
    if not 0 <= left < right <= width or not 0 <= top < bottom <= height:
        raise ValueError("Prediction box lies outside the original image or has no area")
    return tuple(float(v) for v in values)


def load_track_artifact(tracks_path, provenance, recording_id):
    """Verify the complete serialized artifact before selecting an available prefix."""
    from trackrag.vision.classes import EMT_CLASSES

    if provenance.get("recording_id") != recording_id:
        raise ValueError("Saved provenance belongs to another recording")
    if not isinstance(provenance.get("identity_hash"), str) or not re.fullmatch(
            r"[0-9a-f]{64}", provenance["identity_hash"]):
        raise ValueError("Saved provenance requires a valid identity_hash")
    expected_sha = provenance.get("tracks_sha256")
    if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise ValueError("Saved provenance requires a valid tracks_sha256")
    frames = validated_frame_ids(provenance.get("frame_ids"))
    for name, expected in (("frames", len(frames)), ("first_frame", frames[0]), ("last_frame", frames[-1])):
        if name in provenance and (type(provenance[name]) is not int or provenance[name] != expected):
            raise ValueError(f"Provenance {name} disagrees with frame_ids")
    size = provenance.get("first_frame_size")
    if (not isinstance(size, list) or len(size) != 2
            or any(type(v) is not int or v <= 0 for v in size)):
        raise ValueError("Provenance requires positive original image dimensions")
    payload = Path(tracks_path).read_bytes()
    actual_sha = hashlib.sha256(payload).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError("Tracks sha256 differs from provenance")
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    if type(provenance.get("observations")) is not int or len(rows) != provenance["observations"]:
        raise ValueError("Tracks row count differs from provenance['observations']")
    known_frames, seen = set(frames), set()
    for row in rows:
        if (not isinstance(row, dict)
                or row.get("recording_id") != recording_id or row.get("source") != "predicted"
                or type(row.get("frame_id")) is not int or row["frame_id"] not in known_frames
                or type(row.get("track_id")) is not int or row["track_id"] < 0):
            raise ValueError("Track row has a foreign source, recording, frame or invalid identity")
        key = (row["frame_id"], row["track_id"])
        if key in seen:
            raise ValueError("Duplicate track observation in a frame")
        seen.add(key)
        if row.get("class_name") not in EMT_CLASSES:
            raise ValueError("Track row uses an unknown EMT class")
        row["bbox_xyxy"] = validated_box(row.get("bbox_xyxy"), *size)
    return rows, frames, size, actual_sha


def _recording_frame_ids(recording) -> list[int]:
    ids = getattr(recording, "frame_ids", None)
    if ids is None:
        ids = [frame["frame_id"] for frame in recording.iter_frames()]
    return validated_frame_ids(list(ids))


def predict_tracks(recording, *, directory=None, config=None):
    rows, _ = predict_tracks_with_receipt(recording, directory=directory, config=config)
    return rows


def predict_tracks_with_receipt(recording, *, directory=None, config=None):
    """Return validated saved predictions and hashes of the consumed snapshots.

    This is the same route as ``predict_tracks``. It does not run the detector
    or read annotations, and its receipt must not be labelled fresh inference.
    """
    from trackrag.vision.config import load_config

    folder = Path(directory or os.environ.get(SAVED_ENV) or "")
    if not str(directory or os.environ.get(SAVED_ENV) or ""):
        raise ValueError(f"Set {SAVED_ENV} to a folder written by `python -m trackrag.vision predict`")
    recording_id = recording.recording_id
    provenance_path = folder / f"{recording_id}.provenance.json"
    tracks_path = folder / f"{recording_id}.tracks.jsonl"
    if not provenance_path.is_file() or not tracks_path.is_file():
        raise FileNotFoundError(f"No saved predictions for {recording_id} in {folder}")
    provenance_snapshot = provenance_path.read_bytes()
    provenance = json.loads(provenance_snapshot.decode("utf-8"))
    active = config or load_config()
    if provenance.get("identity_hash") != active.identity_hash():
        raise ValueError(f"Saved predictions in {folder} were made with identity {provenance.get('identity_hash')}, "
                         f"but the active vision config {active.name!r} has identity {active.identity_hash()}")
    rows, predicted, size, tracks_sha256 = load_track_artifact(tracks_path, provenance, recording_id)
    available = _recording_frame_ids(recording)
    if predicted[:len(available)] != available:
        raise ValueError(f"{recording_id}: the recording's frames are not a prefix of the predicted frames")
    for frame_id in available:
        frame = recording.get_frame(frame_id)
        if frame["recording_id"] != recording_id or frame["frame_id"] != frame_id:
            raise ValueError("Available frame metadata belongs to another recording or frame")
        if [frame["width"], frame["height"]] != size:
            raise ValueError(f"{recording_id}: frame size differs from prediction provenance")
    last = available[-1]
    selected = [row for row in rows if row["frame_id"] <= last]
    return selected, {
        "kind": "validated_saved_vision_predictions",
        "recording_id": recording_id,
        "vision_identity_hash": active.identity_hash(),
        "tracks_sha256": tracks_sha256,
        "provenance_sha256": hashlib.sha256(provenance_snapshot).hexdigest(),
        "tracks_path": str(tracks_path.resolve()),
        "provenance_path": str(provenance_path.resolve()),
        "prefix_first_frame": available[0],
        "prefix_last_frame": last,
        "prefix_frames": len(available),
        "prefix_observations": len(selected),
        "fresh_detector_inference": False,
    }
