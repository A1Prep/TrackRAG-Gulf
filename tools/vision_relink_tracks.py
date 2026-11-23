"""Apply the frozen online continuity profile to verified saved BoT-SORT tracks.

Writes new tracks and provenance to a separate folder. The detector does not run
and no annotations, images or raw detections are required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from tools.vision_saved_adapters import load_track_artifact
from trackrag.vision.config import CONTINUITY_KIND, PROJECT_ROOT, VisionConfig, load_config
from trackrag.vision.continuity import TrackContinuity


def relink_recording(source: Path, destination: Path, recording: str, config: VisionConfig) -> None:
    """Verify the source profile, replay each observed frame, and record both artifact hashes."""
    if config.tracker.kind != CONTINUITY_KIND:
        raise ValueError(f"Select a target config with tracker kind {CONTINUITY_KIND}")
    parent = config.with_overrides(tracker={"kind": "ultralytics_botsort"})
    provenance_bytes = (source / f"{recording}.provenance.json").read_bytes()
    original = json.loads(provenance_bytes)
    if original.get("identity_hash") != parent.identity_hash() or original.get("identity") != parent.identity():
        raise ValueError(f"{recording}: source detector and tracking settings must match the base BoT-SORT profile")
    started = time.perf_counter()
    rows, frames, _, source_sha = load_track_artifact(source / f"{recording}.tracks.jsonl", original, recording)
    by_frame = defaultdict(list)
    for row in rows:
        by_frame[row["frame_id"]].append(row)
    model = TrackContinuity.frozen_profile()
    identities = set()
    digest = hashlib.sha256()
    with (destination / f"{recording}.tracks.jsonl").open("xb") as stream:
        for frame in frames:
            relabelled = model.relabel_observations(frame, by_frame.get(frame, []))
            for row in sorted(relabelled, key=lambda observation: observation["track_id"]):
                payload = (json.dumps(row, allow_nan=False) + "\n").encode("utf-8")
                stream.write(payload)
                digest.update(payload)
                identities.add(row["track_id"])
    provenance = dict(original)
    # Original timestamps and detector runtime describe the parent run, not the replay.
    for key in ("seconds", "seconds_per_frame", "annotation_access", "packages", "python", "data_root",
                "annotations_dir_present_in_data_root"):
        provenance.pop(key, None)
    provenance.update(
        config_name=config.name, config_source=config.source_path, identity=config.identity(),
        identity_hash=config.identity_hash(), tracker=config.identity()["tracker"], tracks_sha256=digest.hexdigest(),
        identities=len(identities), track_handoffs=model.handoffs, created_utc=datetime.now(timezone.utc).isoformat(),
        annotation_access="Reassociation reads verified saved tracks only.",
        continuity_replay={"source_identity_hash": parent.identity_hash(), "source_tracks_sha256": source_sha,
                           "source_provenance_sha256": hashlib.sha256(provenance_bytes).hexdigest(),
                           "elapsed_seconds": round(time.perf_counter() - started, 4)},
    )
    (destination / f"{recording}.provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--recordings", nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/vision/finetuned-continuity.json")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("Choose a new output directory")
    if len(set(args.recordings)) != len(args.recordings) or any(
            Path(key).name != key or key in (".", "..") for key in args.recordings):
        parser.error("Recordings must be unique recording names")
    config = load_config(args.config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".relink-", dir=args.out.parent) as temp:
        staging = Path(temp) / "predictions"
        staging.mkdir()
        for recording in args.recordings:
            relink_recording(args.pred_dir, staging, recording, config)
        staging.rename(args.out)
    print(f"Saved {len(args.recordings)} recordings to {args.out} (identity {config.identity_hash()[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
