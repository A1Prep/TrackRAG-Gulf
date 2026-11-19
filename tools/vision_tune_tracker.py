"""Replay saved detections through BoT-SORT under a small fixed grid of tracker settings (dev only).

The detector runs once (``python -m trackrag.vision predict`` on the dev recordings). This tool
then re-runs the exact predict_tracks pipeline (same post-processing, ego filter, tracker code,
class vote) with a ReplayDetector that returns those saved detections, once per grid point, and
writes each variant's tracks for ``tools.vision_evaluate``. Only tracker settings change.

    python -m tools.vision_tune_tracker --data-root DATA --recordings video_054907 video_174340 \
        --detections RUN_DIR --config configs/vision/pretrained.json --out TUNE_DIR

Protocol: the grid is fixed in advance, selection is by pooled class-agnostic IDF1 on the dev
recordings, and the chosen settings are frozen before any test run.

With ``--frozen`` only ``--config`` (the frozen settings) is replayed, and the output is a complete
prediction folder (detections, tracks, provenance), the same files ``python -m trackrag.vision
predict`` writes, so a detector run made earlier is re-tracked without running the detector again:

    python -m tools.vision_tune_tracker --frozen --data-root DATA --recordings video_122233 ... \
        --detections TEST_RUN_DIR --config configs/vision/pretrained.json --out FINAL_DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from trackrag.vision import predict_tracks
from trackrag.vision.config import load_config
from trackrag.vision.detections import Detection

# (track_high_thresh = new_track_thresh) x fuse_score. A new track can only start from a
# high-score detection, so high and new thresholds move together.
GRID = [
    {"fuse_score": fuse, "track_high_thresh": threshold, "new_track_thresh": threshold}
    for fuse in (True, False) for threshold in (0.25, 0.2, 0.15)
]


def variant_name(settings: dict) -> str:
    return f"fuse{int(settings['fuse_score'])}_high{settings['track_high_thresh']:.2f}"


class ReplayDetector:
    """Returns the detector outputs saved by a predict run, frame by frame (original pixels)."""

    kind = "replay"

    def __init__(self, detections_path: Path, recording_id: str):
        self.frames: dict[int, list[Detection]] = {}
        self.sizes: dict[int, tuple[int, int]] = {}
        with Path(detections_path).open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["recording_id"] != recording_id:
                    raise ValueError(f"{detections_path} holds another recording")
                frame = int(row["frame_id"])
                self.sizes[frame] = (int(row["width"]), int(row["height"]))
                self.frames[frame] = [Detection(tuple(d["bbox_xyxy"]), d["score"], d["class_name"])
                                      for d in row["native"]]

    def detect_frame(self, frame_id: int, width: int, height: int) -> list[Detection]:
        if frame_id not in self.frames:
            raise KeyError(f"No saved detections for frame {frame_id}")
        if self.sizes[frame_id] != (width, height):
            raise ValueError(f"Frame {frame_id} size differs from the saved detections")
        return self.frames[frame_id]

    def describe(self) -> dict:
        return {"kind": self.kind}


class TracksSink:
    def __init__(self, directory: Path, recording_id: str):
        self.directory, self.recording_id = directory, recording_id

    def frame(self, record: dict) -> None:  # detections are not re-written
        pass

    def finish(self, provenance: dict, observations) -> None:
        with (self.directory / f"{self.recording_id}.tracks.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in observations:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        (self.directory / f"{self.recording_id}.provenance.json").write_text(
            json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def tune(data_root: Path, recordings: list[str], detections_dir: Path, config_path: Path, out: Path,
         max_frames: int | None = None) -> dict:
    from trackrag.emt import open_recording

    if out.exists():
        raise FileExistsError(f"{out} exists; choose a new folder")
    base = load_config(config_path)
    out.mkdir(parents=True)
    index = {"base_config": str(config_path), "base_identity": base.identity_hash(), "grid": GRID,
             "detections_dir": str(detections_dir), "recordings": recordings, "variants": {}}
    for settings in GRID:
        name = variant_name(settings)
        config = base.with_overrides(tracker=settings)
        target = out / name
        target.mkdir()
        for recording_id in recordings:
            recording = open_recording(data_root, recording_id)
            if max_frames:
                from trackrag.vision.__main__ import PrefixRecording
                recording = PrefixRecording(recording, max_frames)
            detector = ReplayDetector(detections_dir / f"{recording_id}.detections.jsonl", recording_id)
            rows = predict_tracks(recording, config=config, detector=detector, sink=TracksSink(target, recording_id))
            print(f"{name} {recording_id}: {len(rows)} observations, "
                  f"{len({r['track_id'] for r in rows})} identities", flush=True)
        index["variants"][name] = {"tracker": settings, "identity_hash": config.identity_hash(), "dir": str(target)}
    (out / "tune_index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index


def frozen(data_root: Path, recordings: list[str], detections_dir: Path, config_path: Path, out: Path,
           max_frames: int | None = None, tracker_factory=None) -> None:
    """Re-track saved detections with one fixed config and write a complete prediction folder.

    The detections file is written again by the same sink as a live run and must equal the saved
    file byte for byte, which shows that the replay reproduced the detector stage exactly.
    """
    import hashlib
    import shutil

    from trackrag.emt import open_recording
    from trackrag.vision.__main__ import JsonlSink, PrefixRecording, _package_versions

    partial = out.with_name(out.name + ".partial")
    if out.exists() or partial.exists():
        raise FileExistsError(f"{out} or {partial} exists; choose a new folder")
    config = load_config(config_path)
    opened = [open_recording(data_root, recording_id) for recording_id in recordings]
    partial.mkdir(parents=True)
    try:
        for recording in opened:
            recording_id = recording.recording_id
            source = detections_dir / f"{recording_id}.detections.jsonl"
            saved = source.read_bytes()
            if max_frames:
                recording = PrefixRecording(recording, max_frames)
            rows = predict_tracks(recording, config=config, detector=ReplayDetector(source, recording_id),
                                  tracker_factory=tracker_factory, sink=JsonlSink(partial, recording_id))
            rewritten = (partial / f"{recording_id}.detections.jsonl").read_bytes()
            reproduced = saved.startswith(rewritten) if max_frames else rewritten == saved
            if not reproduced:
                raise ValueError(f"{recording_id}: replayed detections differ from {source}")
            path = partial / f"{recording_id}.provenance.json"
            provenance = json.loads(path.read_text(encoding="utf-8"))
            provenance.update(data_root=str(data_root), max_frames=max_frames, packages=_package_versions(),
                              replayed_from={"detections": str(source),
                                             "detections_sha256": hashlib.sha256(saved).hexdigest()})
            path.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(f"{recording_id}: {len(rows)} observations, {provenance['identities']} identities", flush=True)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    partial.rename(out)
    print(f"Saved predictions to {out}", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--recordings", nargs="+", required=True)
    parser.add_argument("--detections", type=Path, required=True, help="Folder with <rec>.detections.jsonl")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--frozen", action="store_true",
                        help="Replay only --config and write a complete prediction folder")
    args = parser.parse_args(argv)
    if args.frozen:
        frozen(args.data_root, args.recordings, args.detections, args.config, args.out, args.max_frames)
    else:
        tune(args.data_root, args.recordings, args.detections, args.config, args.out, args.max_frames)
    return 0


if __name__ == "__main__":
    sys.exit(main())
