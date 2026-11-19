r"""Task 1 command line: run predictions on EMT recordings, or draw saved predictions.

    python -m trackrag.vision predict --data-root R --recordings A B --config configs/vision/pretrained.json \
        --weights-dir /content/models --device cuda --out runs/vision/pretrained-test
    python -m trackrag.vision draw --data-root R --recording A --tracks runs/.../A.tracks.jsonl --frames 1 50 100 --out runs/draw
    python -m trackrag.vision draw --data-root R --recording A --tracks ... --track-id 7 --every 5 --out runs/draw

Outputs stay outside Git (runs/ is ignored). Prediction folders hold <rec>.tracks.jsonl
(TrackObservation rows), <rec>.detections.jsonl (per-frame native and mapped detections with
scores) and <rec>.provenance.json (identity hash, checkpoint hash, versions, timing).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


class PrefixRecording:
    """The first ``limit`` frames of a recording (smoke runs). Prefixes are valid inputs because
    predictions are online."""

    def __init__(self, recording, limit: int):
        self._recording = recording
        self.recording_id = recording.recording_id
        ids = getattr(recording, "frame_ids", None)
        if ids is None:
            ids = [frame["frame_id"] for frame in recording.iter_frames()]
        self.frame_ids = list(ids)[:limit]

    def iter_frames(self):
        for frame_id in self.frame_ids:
            yield self._recording.get_frame(frame_id)

    def get_frame(self, frame_id):
        if frame_id not in self.frame_ids:
            raise KeyError(frame_id)
        return self._recording.get_frame(frame_id)


class JsonlSink:
    def __init__(self, directory: Path, recording_id: str):
        self.directory, self.recording_id = directory, recording_id
        self._detections = (directory / f"{recording_id}.detections.jsonl").open("w", encoding="utf-8", newline="\n")

    def frame(self, record: dict) -> None:
        self._detections.write(json.dumps(record, allow_nan=False) + "\n")

    def finish(self, provenance: dict, observations) -> None:
        import hashlib

        self._detections.close()
        tracks = self.directory / f"{self.recording_id}.tracks.jsonl"
        with tracks.open("w", encoding="utf-8", newline="\n") as out:
            for row in observations:
                out.write(json.dumps(row, allow_nan=False) + "\n")
        provenance = dict(provenance, tracks_sha256=hashlib.sha256(tracks.read_bytes()).hexdigest())
        (self.directory / f"{self.recording_id}.provenance.json").write_text(
            json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _package_versions() -> dict:
    from importlib import metadata

    versions = {}
    for package in ("rfdetr", "ultralytics", "torch", "torchvision", "numpy", "pillow", "supervision",
                    "opencv-python", "opencv-python-headless"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    try:
        import cv2

        versions["cv2_runtime"] = cv2.__version__
    except Exception as error:  # pragma: no cover - environment report only
        versions["cv2_runtime"] = repr(error)
    return versions


def _predict(args) -> int:
    from trackrag.emt import open_recording

    from . import predict_tracks
    from .config import load_config
    from .detector import RFDETRDetector

    import shutil

    out = Path(args.out)
    partial = out.with_name(out.name + ".partial")
    if out.exists():
        raise SystemExit(f"Output folder already exists: {out} (choose a new one)")
    if partial.exists():
        raise SystemExit(f"A previous run left {partial}; delete it, then run again")
    config = load_config(args.config, weights_dir=args.weights_dir, device=args.device)
    if args.fuse_score is not None:
        config = config.with_overrides(fuse_score=args.fuse_score == "true")
    recordings = [open_recording(args.data_root, recording_id) for recording_id in args.recordings]  # fail fast on typos
    detector = RFDETRDetector(config)  # load once for all recordings
    versions = _package_versions()
    partial.mkdir(parents=True, exist_ok=False)
    annotations_present = (Path(args.data_root) / "annotations").exists()
    try:
        for recording in recordings:
            recording_id = recording.recording_id
            if args.max_frames:
                recording = PrefixRecording(recording, args.max_frames)
            sink = JsonlSink(partial, recording_id)
            predict_tracks(recording, config=config, detector=detector, sink=sink)
            path = partial / f"{recording_id}.provenance.json"
            provenance = json.loads(path.read_text(encoding="utf-8"))
            provenance.update(data_root=str(args.data_root), annotations_dir_present_in_data_root=annotations_present,
                              max_frames=args.max_frames, packages=versions)
            path.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(f"{recording_id}: {provenance['frames']} frames, {provenance['observations']} observations, "
                  f"{provenance['identities']} identities, {provenance['seconds_per_frame']} s/frame", flush=True)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    partial.rename(out)
    print(f"Saved predictions to {out}", flush=True)
    return 0


def _draw(args) -> int:
    from PIL import Image, ImageDraw

    from trackrag.emt import open_recording

    rows = [json.loads(line) for line in Path(args.tracks).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row["recording_id"] == args.recording]
    by_frame: dict[int, list] = {}
    for row in rows:
        by_frame.setdefault(row["frame_id"], []).append(row)
    recording = open_recording(args.data_root, args.recording)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.track_id is not None:
        frames = sorted(row["frame_id"] for row in rows if row["track_id"] == args.track_id)
        if not frames:
            raise SystemExit(f"Track {args.track_id} does not occur in {args.tracks}")
        frames = frames[::max(1, args.every)][: args.max_images]
    else:
        frames = args.frames
    crops = []
    for frame_id in frames:
        frame = recording.get_frame(frame_id)
        with Image.open(frame["image_path"]) as opened:
            canvas = opened.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        line = max(2, canvas.width // 640)
        for row in by_frame.get(frame_id, []):
            highlight = args.track_id is None or row["track_id"] == args.track_id
            color = (255, 0, 0) if highlight else (255, 200, 0)
            left, top, right, bottom = row["bbox_xyxy"]
            draw.rectangle((left, top, right, bottom), outline=color, width=line)
            draw.text((left, max(0, top - 12 * line)), f"{row['track_id']} {row['class_name']}", fill=color)
            if args.track_id is not None and row["track_id"] == args.track_id:
                margin = max(right - left, bottom - top) * 0.5
                crop = canvas.crop((max(0, left - margin), max(0, top - margin),
                                    min(canvas.width, right + margin), min(canvas.height, bottom + margin)))
                crop = crop.resize((max(1, round(crop.width * 160 / crop.height)), 160))
                ImageDraw.Draw(crop).text((2, 2), f"f{frame_id}", fill=(255, 255, 255))
                crops.append(crop)
        scale = min(1.0, args.width / canvas.width)
        if scale < 1.0:
            canvas = canvas.resize((round(canvas.width * scale), round(canvas.height * scale)))
        name = f"{args.recording}_{frame_id:05d}" + (f"_track{args.track_id}" if args.track_id is not None else "")
        canvas.save(out / f"{name}.jpg", quality=85)
    if crops:
        strip = Image.new("RGB", (sum(c.width for c in crops) + 4 * (len(crops) - 1), 160), (0, 0, 0))
        x = 0
        for crop in crops:
            strip.paste(crop, (x, 0))
            x += crop.width + 4
        strip.save(out / f"{args.recording}_track{args.track_id}_strip.jpg", quality=85)
    print(f"Wrote {len(frames)} images" + (" and a track strip" if crops else "") + f" to {out}", flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m trackrag.vision", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict", help="Run detector + tracker on whole recordings")
    predict.add_argument("--data-root", required=True)
    predict.add_argument("--recordings", nargs="+", required=True)
    predict.add_argument("--config", help="Config JSON (default: TRACKRAG_VISION_CONFIG or configs/vision/pretrained.json)")
    predict.add_argument("--weights-dir", default=os.environ.get("TRACKRAG_VISION_WEIGHTS_DIR"))
    predict.add_argument("--device", choices=("cuda", "cpu"))
    predict.add_argument("--fuse-score", choices=("true", "false"), help="Override tracker fuse_score (dev check only)")
    predict.add_argument("--max-frames", type=int, help="Only the first N frames (smoke runs)")
    predict.add_argument("--out", required=True)
    draw = commands.add_parser("draw", help="Draw saved predictions on original frames")
    draw.add_argument("--data-root", required=True)
    draw.add_argument("--recording", required=True)
    draw.add_argument("--tracks", required=True)
    draw.add_argument("--frames", nargs="+", type=int)
    draw.add_argument("--track-id", type=int)
    draw.add_argument("--every", type=int, default=5)
    draw.add_argument("--max-images", type=int, default=12)
    draw.add_argument("--width", type=int, default=1280)
    draw.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "predict":
        if args.max_frames is not None and args.max_frames < 1:
            parser.error("--max-frames must be positive")
        return _predict(args)
    if (args.frames is None) == (args.track_id is None):
        parser.error("draw needs exactly one of --frames or --track-id")
    return _draw(args)


if __name__ == "__main__":
    sys.exit(main())
