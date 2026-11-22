"""Build image-bound grounding inputs from an exact frame and predicted tracks.

This module never reads human annotations or supplies destination labels. The
caller remains responsible for validating saved-prediction/model provenance;
the existing Task 1 saved-artifact loader performs that check.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re
import warnings

from PIL import Image, UnidentifiedImageError

from trackrag.contracts import Recording, TrackObservation
from trackrag.dropoff.contracts import InstructionRequest, validate_box, validate_request


MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_ORIGINAL_PIXELS = 48_000_000
MAX_OBSERVATIONS = 250_000
_TRACK_FIELDS = frozenset((
    "recording_id", "frame_id", "track_id", "class_name", "bbox_xyxy", "source",
))


def _validate_observation(row, recording_id, width, height):
    if not isinstance(row, dict) or set(row) != _TRACK_FIELDS:
        raise ValueError("predicted observation has missing or unknown fields")
    if row["recording_id"] != recording_id or row["source"] != "predicted":
        raise ValueError("observation recording/source differs from the predicted scene")
    for name in ("frame_id", "track_id"):
        if type(row[name]) is not int or row[name] < 0:
            raise ValueError(f"observation {name} must be a nonnegative integer")
    label = row["class_name"]
    if not isinstance(label, str) or not label.strip() or len(label) > 100:
        raise ValueError("observation class_name must be bounded nonempty text")
    validate_box(row["bbox_xyxy"], width, height, "observation bbox")


def _inspect_image(frame):
    path = Path(frame["image_path"]).expanduser().resolve()
    # Read once, bounded even if the file changes after stat/header inspection.
    with path.open("rb") as stream:
        payload = stream.read(MAX_IMAGE_BYTES + 1)
    if not payload or len(payload) > MAX_IMAGE_BYTES:
        raise ValueError("selected image is empty or exceeds the image byte budget")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as image:
                if image.width * image.height > MAX_ORIGINAL_PIXELS:
                    raise ValueError("selected image exceeds the original-pixel budget")
                if image.size != (frame["width"], frame["height"]):
                    raise ValueError("selected image dimensions differ from original frame metadata")
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("selected frame must contain one image, not an animation")
                image.load()
    except (UnidentifiedImageError, Image.DecompressionBombError,
            Image.DecompressionBombWarning, OSError, SyntaxError) as error:
        raise ValueError("selected image cannot be decoded within the image bounds") from error
    return path, sha256(payload).hexdigest()


def build_instruction_request(
    recording: Recording,
    observations: Iterable[TrackObservation],
    *,
    frame_id: int,
    request_id: str,
    text: str,
    input_mode: str = "typed",
    dialogue: list[dict] | None = None,
    original_transcript: str | None = None,
    session_id: str | None = None,
    provenance: dict | None = None,
) -> InstructionRequest:
    """Return a validated request containing only this frame's predictions.

    ``get_frame`` must return the exact requested original frame. Missing frames
    retain that reader's ``KeyError``; malformed metadata/observations raise
    ``ValueError`` and unavailable image files raise ``OSError``. All supplied
    observations are checked before temporal selection, so a foreign or corrupt
    row cannot disappear silently. Boxes must use this recording's original
    image dimensions, matching the strict Task 1 saved-prediction adapter.

    Only current-frame observations enter the request. Past/future rows, human
    labels, masks and inferred destination coordinates are never added. A hash
    in supplied provenance is an expected value and cannot be silently replaced.
    Inputs are copied; this function never changes reader/caller records.
    """
    if type(frame_id) is not int or frame_id < 0:
        raise ValueError("frame_id must be a nonnegative integer")
    recording_id = recording.recording_id
    if not isinstance(recording_id, str) or not recording_id.strip():
        raise ValueError("recording_id must be nonempty text")
    frame = deepcopy(recording.get_frame(frame_id))
    if not isinstance(frame, dict):
        raise ValueError("get_frame must return an original Frame record")
    if frame.get("recording_id") != recording_id or frame.get("frame_id") != frame_id:
        raise ValueError("get_frame returned a different recording or original frame")
    if provenance is not None and not isinstance(provenance, dict):
        raise ValueError("scene provenance must be an object")
    metadata = deepcopy(provenance) if provenance is not None else {}
    expected_sha = metadata.get("image_sha256")
    if "image_sha256" in metadata and (
        not isinstance(expected_sha, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None
    ):
        raise ValueError("expected image_sha256 must be a lowercase SHA256")
    request = {
        "request_id": request_id,
        "scene": {"frame": frame, "source": "predicted", "tracks": [], "provenance": metadata},
        "text": text, "input_mode": input_mode,
        "dialogue": deepcopy(dialogue) if dialogue is not None else [],
    }
    if original_transcript is not None:
        request["original_transcript"] = original_transcript
    if session_id is not None:
        request["session_id"] = session_id
    # Validate IDs, dimensions, text, dialogue and JSON metadata before file IO.
    request = validate_request(request)
    frame = request["scene"]["frame"]
    path, actual_sha = _inspect_image(frame)
    if expected_sha is not None and actual_sha != expected_sha:
        raise ValueError("selected image SHA256 differs from expected provenance")
    frame["image_path"] = str(path)
    request["scene"]["provenance"]["image_sha256"] = actual_sha
    if isinstance(observations, (str, bytes, dict)) or not isinstance(observations, Iterable):
        raise ValueError("observations must be a bounded iterable of predicted records")
    current = []
    for index, row in enumerate(observations):
        if index >= MAX_OBSERVATIONS:
            raise ValueError("observations exceed the preparation row budget")
        _validate_observation(row, recording_id, frame["width"], frame["height"])
        if row["frame_id"] == frame_id:
            if len(current) >= 512:
                raise ValueError("current frame exceeds the 512-observation input budget")
            current.append(deepcopy(row))
    request["scene"]["tracks"] = current
    return validate_request(request)
