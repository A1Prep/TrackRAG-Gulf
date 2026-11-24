"""Task 2 reader for EMT's split-first images/annotations layout."""
from __future__ import annotations
import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional
import cv2
from trackrag.contracts import Frame, Recording, TrackObservation

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_FRAME_RATE = 10.0
DEFAULT_MANIFEST_PATH = Path("splits") / "task2_manifest.json"
VEHICLE_CLASSES = {"Motorbike", "Small_motorised_vehicle", "Medium_vehicle", "Large_vehicle", "Car", "Bus", "Emergency_vehicle"}
SPLITS = ("train", "test")


class FrameNotFoundError(KeyError):
    """The requested original frame is absent."""


class MissingLabelsError(FileNotFoundError):
    """Labels were requested but the recording's label file is absent."""


@dataclass
class EMTRecording:
    recording_id: str
    data_root: Path
    split: str
    image_dir: Path
    frame_ids: list[int]
    frame_rate: float
    width: int
    height: int
    frame_index: dict[int, Path]
    headers_only: bool = False

    @property
    def first_frame_id(self) -> int:
        return self.frame_ids[0]

    @property
    def annotation_path(self) -> Path:
        return self.data_root / "annotations" / self.split / f"{self.recording_id}.txt"

    def iter_frames(self) -> Iterator[Frame]:
        """Yield shared Frame dictionaries in ascending numerical frame order."""
        for frame_id in self.frame_ids:
            yield self.get_frame(frame_id)

    def get_frame(self, frame_id: int) -> Frame:
        """Return exactly the requested original frame; never substitute a neighbor.

        The recording clock is anchored at ``first_frame_id`` (the first available
        image frame). At ``frame_rate`` FPS, ``timestamp_s`` is
        ``(frame_id - first_frame_id) / frame_rate``. In EMT the first image frame
        is normally also the first annotated frame, but the reader does not assume
        that: an annotation that starts later simply yields timestamps > 0.
        """
        if frame_id not in self.frame_index:
            raise FrameNotFoundError(f"Frame {frame_id} is absent from recording {self.recording_id!r}.")
        path = self.frame_index[frame_id]
        width, height = _frame_size(path, self.headers_only)
        return Frame(
            recording_id=self.recording_id,
            frame_id=frame_id,
            timestamp_s=(frame_id - self.first_frame_id) / self.frame_rate,
            width=width,
            height=height,
            image_path=str(path),
        )


def _frame_size(path: Path, headers_only: bool) -> tuple[int, int]:
    if headers_only:
        from PIL import Image

        try:
            with Image.open(path) as original:
                return original.size
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot read original frame header: {path}") from error
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode original frame image: {path}")
    height, width = image.shape[:2]
    return width, height


def _candidate_image_dirs(data_root: Path, split: str, recording_id: str) -> list[Path]:
    base = data_root / "images" / split
    return [base / "images" / recording_id, base / recording_id]


def find_image_dir(data_root: Path, split: str, recording_id: str) -> Optional[Path]:
    return next((p for p in _candidate_image_dirs(data_root, split, recording_id) if p.is_dir()), None)


def _index_frames(image_dir: Path) -> dict[int, Path]:
    index: dict[int, Path] = {}
    for path in image_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS or not path.stem.isdigit():
            continue
        frame_id = int(path.stem)
        if frame_id in index:
            raise ValueError(f"Duplicate frame {frame_id}: {index[frame_id].name} and {path.name}")
        index[frame_id] = path
    return index


def read_release_frame_rate(data_root: Path | str) -> Optional[float]:
    """Return the release's declared frame rate if ``annotations/metadata.txt`` exposes it.

    EMT publishes images and annotations separately, so a metadata file is *not*
    required. When it is absent this returns ``None`` and the caller falls back to
    ``DEFAULT_FRAME_RATE`` (documented as the assumed release rate).
    """
    path = Path(data_root) / "annotations" / "metadata.txt"
    if not path.is_file():
        return None
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        for key in ("fps", "frame_rate", "framerate", "frame rate"):
            match = re.match(rf"^{re.escape(key)}\s*[:=]\s*(.*)$", line, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                except ValueError as exc:
                    raise ValueError(f"Invalid frame rate in {path}: {line}") from exc
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"Frame rate in {path} must be finite and positive")
                return value
    return None


def open_recording(data_root: Path | str, recording_id: str, frame_rate: Optional[float] = None,
                   split: Optional[str] = None, *, headers_only: bool = False) -> EMTRecording:
    """Open an EMT recording. Use headers_only when the caller decodes its selected evidence separately."""
    data_root = Path(data_root)
    if frame_rate is None:
        frame_rate = read_release_frame_rate(data_root) or DEFAULT_FRAME_RATE
    if not math.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError("frame_rate must be finite and positive")
    splits = [split] if split is not None else list(SPLITS)
    if any(s not in SPLITS for s in splits):
        raise ValueError("split must be train, test, or None")
    found = [(s, find_image_dir(data_root, s, recording_id)) for s in splits]
    found = [(s, p) for s, p in found if p is not None]
    if not found:
        raise FileNotFoundError(f"No image folder for {recording_id!r} under {data_root / 'images'}")
    if len(found) > 1:
        raise ValueError(f"Recording {recording_id!r} occurs in multiple splits: {[s for s, _ in found]}")
    recording_split, image_dir = found[0]
    frame_index = _index_frames(image_dir)
    if not frame_index:
        raise ValueError(f"Recording {recording_id!r} has no numerically named images")
    frame_ids = sorted(frame_index)
    width, height = _frame_size(frame_index[frame_ids[0]], headers_only)
    return EMTRecording(recording_id, data_root, recording_split, image_dir, frame_ids, float(frame_rate),
                        width, height, frame_index, headers_only)


def load_annotated_tracks(recording: Recording, annotation_path: Optional[Path | str] = None) -> list[TrackObservation]:
    """Load original GMOT-style xyxy boxes as shared TrackObservation dictionaries.

    Column layout assumed by the EMT release (whitespace- or comma-separated):

        frame_id  track_id  class_name  <x>  <y>  <z>  left  top  right  bottom

    Columns ``6..9`` are original-pixel ``left, top, right, bottom`` (xyxy). Rows
    whose ``class_name`` is ``AV`` (ego vehicle) or whose ``frame_id`` has no
    matching image are skipped. A missing label file raises ``MissingLabelsError``
    so the recording can still be opened in images-only mode.
    """
    split = getattr(recording, "split")
    default = recording.data_root / "annotations" / split / f"{recording.recording_id}.txt"
    path = Path(annotation_path) if annotation_path is not None else default
    if not path.is_file() and annotation_path is None:
        alternate = default.with_suffix("")
        if alternate.is_file():
            path = alternate
    if not path.is_file():
        raise MissingLabelsError(f"No labels for {recording.recording_id!r}: {path}. Images-only mode remains available.")
    known_frames = set(recording.frame_ids)
    observations: list[TrackObservation] = []
    seen = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"[,\s]+", line)
            try:
                numbers = (float(parts[0]), float(parts[1]))
                if any(not math.isfinite(value) or value < 0 or not value.is_integer() for value in numbers):
                    raise ValueError("Frame and track IDs must be nonnegative integers")
                frame_id, track_id = map(int, numbers)
                class_name = parts[2]
                left, top, right, bottom = map(float, parts[6:10])
            except (IndexError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid GMOT row: {raw!r}") from exc
            if class_name == "AV" or frame_id not in known_frames:
                continue
            if not all(math.isfinite(value) for value in (left, top, right, bottom)) or right <= left or bottom <= top:
                raise ValueError(f"{path}:{line_number}: box must have finite coordinates and positive area")
            identity = (frame_id, track_id)
            if identity in seen:
                raise ValueError(f"{path}:{line_number}: duplicate track {track_id} in frame {frame_id}")
            seen.add(identity)
            observations.append(TrackObservation(
                recording_id=recording.recording_id, frame_id=frame_id, track_id=track_id,
                class_name=class_name, bbox_xyxy=(left, top, right, bottom), source="annotated"))
    return sorted(observations, key=lambda o: (o["frame_id"], o["track_id"]))


def recording_ids_for_split(data_root: Path | str, split: str) -> list[str]:
    """List whole recordings represented by image folders in train or test."""
    if split not in SPLITS:
        raise ValueError("split must be train or test")
    base = Path(data_root) / "images" / split
    container = base / "images" if (base / "images").is_dir() else base
    if not container.is_dir():
        return []
    return sorted(p.name for p in container.iterdir() if p.is_dir() and _index_frames(p))


def assert_splits_disjoint(manifest: dict[str, list[str]], available_recording_ids: Optional[set[str]] = None) -> None:
    """Require exactly train/dev/test keys and no duplicate recording membership."""
    if set(manifest) != {"train", "dev", "test"}:
        raise ValueError("Manifest keys must be exactly train, dev, and test")
    owner: dict[str, str] = {}
    for split, ids in manifest.items():
        for recording_id in ids:
            if recording_id in owner:
                raise ValueError(f"Recording {recording_id!r} occurs in {owner[recording_id]} and {split}")
            owner[recording_id] = split
    if available_recording_ids is not None:
        listed = set(owner)
        missing = sorted(available_recording_ids - listed); extra = sorted(listed - available_recording_ids)
        if missing or extra:
            raise ValueError(f"Manifest/disk mismatch; missing={missing}, extra={extra}")


def make_split_manifest(data_root: Path | str, dev_count: int = 2, seed: int = 42, output_path: Optional[Path | str] = None) -> dict[str, list[str]]:
    """Keep every official test recording unchanged; select dev only from train.

    The manifest is written *outside* the downloaded footage tree by default so it
    can be committed: ``splits/task2_manifest.json`` relative to the repository.
    """
    if dev_count < 0:
        raise ValueError("dev_count must be non-negative")
    data_root = Path(data_root)
    official_train = recording_ids_for_split(data_root, "train")
    official_test = recording_ids_for_split(data_root, "test")
    overlap = set(official_train) & set(official_test)
    if overlap:
        raise ValueError(f"Recordings occur in both official train and test: {sorted(overlap)}")
    count = min(dev_count, max(0, len(official_train) - 1))
    chosen = set(random.Random(seed).sample(official_train, count)) if count else set()
    manifest = {"train": [r for r in official_train if r not in chosen], "dev": sorted(chosen), "test": official_test}
    assert_splits_disjoint(manifest, set(official_train) | set(official_test))
    output_path = Path(output_path) if output_path is not None else DEFAULT_MANIFEST_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def inspect_recording(data_root: Path | str, recording_id: str) -> None:
    """Print first/last frame details and one labeled object's original box."""
    recording = open_recording(data_root, recording_id)
    first = recording.get_frame(recording.frame_ids[0]); last = recording.get_frame(recording.frame_ids[-1])
    for label, frame in (("FIRST", first), ("LAST", last)):
        print(f"{label}: recording={frame['recording_id']} split={recording.split} frame={frame['frame_id']} timestamp_s={frame['timestamp_s']:.3f} dimensions={frame['width']}x{frame['height']} image_path={frame['image_path']}")
    try:
        observations = load_annotated_tracks(recording)
    except MissingLabelsError as exc:
        print(f"IMAGES ONLY: {exc}"); return
    if not observations:
        print("LABELS: no target observations reference available frames"); return
    chosen = next((o for o in observations if o["class_name"] in VEHICLE_CLASSES), observations[0])
    print(f"LABELED OBJECT: recording={chosen['recording_id']} frame={chosen['frame_id']} track={chosen['track_id']} class={chosen['class_name']} bbox_xyxy={chosen['bbox_xyxy']} source={chosen['source']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=inspect_recording.__doc__)
    parser.add_argument("--data-root", required=True, type=Path); parser.add_argument("--recording-id", required=True)
    args = parser.parse_args(); inspect_recording(args.data_root, args.recording_id)
