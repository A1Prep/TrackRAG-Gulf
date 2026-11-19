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

import json
import os
from pathlib import Path

from trackrag.answering import answer_question  # noqa: F401  (re-exported real Task 3 answerer)
from trackrag.emt import open_recording  # noqa: F401  (re-exported real Task 2 reader)
from trackrag.questions import load_questions  # noqa: F401  (re-exported real Task 3 loader)
from trackrag.retrieval import build_index, rank_evidence  # noqa: F401  (re-exported real Task 4)

SAVED_ENV = "TRACKRAG_SAVED_PREDICTIONS"


def _recording_frame_ids(recording) -> list[int]:
    ids = getattr(recording, "frame_ids", None)
    if ids is None:
        ids = [frame["frame_id"] for frame in recording.iter_frames()]
    return [int(frame_id) for frame_id in ids]


def predict_tracks(recording, *, directory=None, config=None):
    from trackrag.vision.config import load_config

    folder = Path(directory or os.environ.get(SAVED_ENV) or "")
    if not str(directory or os.environ.get(SAVED_ENV) or ""):
        raise ValueError(f"Set {SAVED_ENV} to a folder written by `python -m trackrag.vision predict`")
    recording_id = recording.recording_id
    provenance_path = folder / f"{recording_id}.provenance.json"
    tracks_path = folder / f"{recording_id}.tracks.jsonl"
    if not provenance_path.is_file() or not tracks_path.is_file():
        raise FileNotFoundError(f"No saved predictions for {recording_id} in {folder}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    active = config or load_config()
    if provenance.get("identity_hash") != active.identity_hash():
        raise ValueError(f"Saved predictions in {folder} were made with identity {provenance.get('identity_hash')}, "
                         f"but the active vision config {active.name!r} has identity {active.identity_hash()}")
    if provenance.get("recording_id") != recording_id:
        raise ValueError("Saved provenance belongs to another recording")
    predicted = [int(frame_id) for frame_id in provenance["frame_ids"]]
    available = _recording_frame_ids(recording)
    if not available or predicted[: len(available)] != available:
        span = f"{predicted[0]}-{predicted[-1]}" if predicted else "none"
        raise ValueError(f"{recording_id}: the recording's frames are not a prefix of the predicted frames: "
                         f"predictions cover frames {span} ({len(predicted)} frames) but the recording has "
                         f"{len(available)} frames from {available[:1]}")
    payload = tracks_path.read_bytes()
    expected_sha = provenance.get("tracks_sha256")
    if expected_sha is not None:
        import hashlib

        if hashlib.sha256(payload).hexdigest() != expected_sha:
            raise ValueError(f"{tracks_path} changed after it was written (sha256 differs from provenance)")
    if sum(1 for line in payload.decode("utf-8").splitlines() if line.strip()) != provenance.get("observations"):
        raise ValueError(f"{tracks_path} row count differs from provenance['observations'] (incomplete file?)")
    first = recording.get_frame(available[0])
    if [first["width"], first["height"]] != provenance.get("first_frame_size"):
        raise ValueError(f"{recording_id}: frame size differs from the size the predictions were made on")
    last = available[-1]
    rows = []
    for line in tracks_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("recording_id") != recording_id or row.get("source") != "predicted":
            raise ValueError("Saved predictions mix recordings or sources")
        if row["frame_id"] <= last:
            row["bbox_xyxy"] = tuple(float(value) for value in row["bbox_xyxy"])
            rows.append(row)
    return rows
