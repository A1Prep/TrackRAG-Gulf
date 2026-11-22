"""Input-preserving controls for black-box retrieval checks, not accuracy metrics."""

import json
import math
from copy import deepcopy
from pathlib import Path

from PIL import Image, ImageDraw

from .run_audit import _box, integer
from trackrag.contracts import RankedInterval
from trackrag.review import _validate_interval


class FrameView:
    """Implement the shared Recording protocol, without exposing private paths."""

    def __init__(self, recording_id, frames, *, cutoff=None, replacements=None):
        self.recording_id = recording_id
        self._frames = {}
        for frame in frames:
            if cutoff is not None and frame["frame_id"] > cutoff:
                continue
            value = deepcopy(frame)
            if replacements and frame["frame_id"] in replacements:
                value["image_path"] = str(replacements[frame["frame_id"]])
            self._frames[frame["frame_id"]] = value

    def iter_frames(self):
        return (deepcopy(self._frames[key]) for key in sorted(self._frames))

    def get_frame(self, frame_id):
        return deepcopy(self._frames[frame_id])


def checked_frames(recording):
    frames = [deepcopy(frame) for frame in recording.iter_frames()]
    if not frames:
        raise ValueError("The recording has no frames")
    ids = []
    for frame in frames:
        key = integer(frame["frame_id"], "frame_id")
        ids.append(key)
        if frame["recording_id"] != recording.recording_id:
            raise ValueError("Frame metadata mixes recordings")
        for field in ("width", "height"):
            if type(frame[field]) is not int or frame[field] <= 0:
                raise ValueError("Frame dimensions must be positive integers")
        stamp = frame["timestamp_s"]
        if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
            raise ValueError("Frame timestamp must be finite")
        with Image.open(frame["image_path"]) as image:
            if image.size != (frame["width"], frame["height"]):
                raise ValueError("Frame metadata differs from original image dimensions")
            image.verify()
        looked_up = recording.get_frame(key)
        if looked_up != frame:
            raise ValueError("iter_frames and get_frame disagree")
    if ids != sorted(set(ids)):
        raise ValueError("Frames must be unique and in ascending original-ID order")
    return frames


def checked_observations(rows, frames, recording_id, source):
    rows = deepcopy(list(rows))
    lookup = {frame["frame_id"]: frame for frame in frames}
    seen = set()
    for row in rows:
        if row["recording_id"] != recording_id or row["source"] != source:
            raise ValueError("Observations mix recordings or track sources")
        key = (integer(row["track_id"], "track_id"), integer(row["frame_id"], "frame_id"))
        if key in seen:
            raise ValueError("Duplicate track/frame observation")
        seen.add(key)
        frame = lookup.get(row["frame_id"])
        if frame is None:
            raise ValueError("Observation refers to an unavailable frame")
        if not isinstance(row["class_name"], str) or not row["class_name"].strip():
            raise ValueError("Observation class must be a nonempty string")
        _box(row["bbox_xyxy"])
        if row["bbox_xyxy"][2] > frame["width"] or row["bbox_xyxy"][3] > frame["height"]:
            raise ValueError("Observation box exceeds original image dimensions")
    # The ordinary reviewer serializes observations too. Reject unsupported types
    # and nonfinite optional fields rather than silently repairing a handoff.
    json.dumps(rows, allow_nan=False)
    return rows


def checked_ranking(ranked, question, source, observations, frames):
    ranked = deepcopy(list(ranked))
    if len(ranked) > 3:
        raise ValueError("Retriever returned more than three intervals")
    frame_ids = {frame["frame_id"] for frame in frames}
    observed = {}
    for row in observations:
        observed.setdefault(row["track_id"], set()).add(row["frame_id"])
    for rank, interval in enumerate(ranked, 1):
        if not isinstance(interval, dict) or set(interval) != set(RankedInterval.__required_keys__):
            raise ValueError("Ranked intervals require exactly the shared public fields")
        _validate_interval(interval, question, source, rank)
        if interval["track_id"] is None:
            if interval["start_frame"] != interval["end_frame"] or interval["start_frame"] not in frame_ids:
                raise ValueError("Scene evidence requires one available exact frame")
        elif not any(interval["start_frame"] <= frame <= interval["end_frame"]
                     for frame in observed.get(interval["track_id"], ())):
            raise ValueError("Retrieved track has no observation in its interval")
    json.dumps(ranked, allow_nan=False)
    return ranked


def same_ranking(left, right, tolerance):
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if {k: v for k, v in a.items() if k != "score"} != {k: v for k, v in b.items() if k != "score"}:
            return False
        if not math.isclose(a["score"], b["score"], rel_tol=tolerance, abs_tol=tolerance):
            return False
    return True


def changed_future_observations(rows, frames, cutoff, recording_id, source):
    """Change only future labels/positions and introduce one future-only track."""
    lookup = {frame["frame_id"]: frame for frame in frames}
    changed = deepcopy(rows)
    for row in changed:
        if row["frame_id"] <= cutoff:
            continue
        frame = lookup[row["frame_id"]]
        left, top, right, bottom = row["bbox_xyxy"]
        row["bbox_xyxy"] = [frame["width"] - right, frame["height"] - bottom,
                            frame["width"] - left, frame["height"] - top]
        row["class_name"] = "Bus" if row["class_name"] != "Bus" else "Car"
    future = next(frame for frame in frames if frame["frame_id"] > cutoff)
    changed.append({"recording_id": recording_id, "source": source,
                    "track_id": max((r["track_id"] for r in rows), default=-1) + 1,
                    "frame_id": future["frame_id"], "class_name": "Bus",
                    "bbox_xyxy": [0, 0, future["width"] / 2, future["height"] / 2]})
    return checked_observations(changed, frames, recording_id, source)


def replacement_frames(frames, cutoff, directory):
    """Use visibly synthetic future pixels; original footage is never edited."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sizes, replacements = {}, {}
    for frame in frames:
        if frame["frame_id"] <= cutoff:
            continue
        size = (frame["width"], frame["height"])
        if size not in sizes:
            path = directory / f"synthetic-future-{size[0]}x{size[1]}.png"
            image = Image.new("RGB", size, (255, 0, 255))
            draw = ImageDraw.Draw(image)
            for y in range(0, size[1], 64):
                for x in range(0, size[0], 64):
                    if (x // 64 + y // 64) % 2:
                        draw.rectangle((x, y, x + 63, y + 63), fill=(0, 0, 0))
            image.save(path)
            sizes[size] = path
        replacements[frame["frame_id"]] = sizes[size]
    return replacements
