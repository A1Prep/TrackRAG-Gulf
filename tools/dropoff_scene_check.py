r"""Real-data check of the drop-off scene adapter (model environment; overlays and fresh runs go outside Git).

For each vision profile and each (recording, frame): build the scene from the shared saved predictions and validate
it against those predictions, the image and the frame-hash manifest; run the detector on that single frame and
compare its raw output with the saved detection row of the run that made the shared tracks (both directions: every
detection on either side must have a partner of the same class with IoU >= 0.95); draw the scene's own observations
on the original frame. Optionally compare a fresh-inference scene with the saved scene at the same frame (same
classes and geometry; track numbers may differ between runs).

    python -m tools.dropoff_scene_check --data-root R --frame-manifest M --run-manifest S --weights-dir W --device cpu \
        --profile finetuned=configs/vision/finetuned.json=PRED_FT --profile pretrained=configs/vision/pretrained.json=PRED_PT \
        --detections-dirs RUN1 RUN2 ... --frames video_054604:135 video_204647:1200 video_155425:600 \
        --fresh finetuned:video_054604:135 --out-dir OUT --record experiments/dropoff/scene_handoff_check_20261120.json \
        --example finetuned:video_054604:135=experiments/dropoff/scene_example_video_054604_f135.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from trackrag.dropoff import scenes
from trackrag.emt import open_recording
from trackrag.vision.geometry import iou

MATCH_IOU = 0.95


class OneFrame:
    """A recording that contains only one frame (the detector is stateless, so this reproduces that frame's row)."""

    def __init__(self, recording, frame_id):
        self.recording_id, self._recording, self.frame_ids = recording.recording_id, recording, [frame_id]

    def iter_frames(self):
        yield self._recording.get_frame(self.frame_ids[0])

    def get_frame(self, frame_id):
        if frame_id != self.frame_ids[0]:
            raise KeyError(frame_id)
        return self._recording.get_frame(frame_id)


def _detection_row(path: Path, frame_id: int) -> dict:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["frame_id"] == frame_id:
                return row
    raise ValueError(f"{path} has no row for frame {frame_id}")


def _find_run(dirs, recording_id: str, tracks_sha: str):
    """The run folder whose tracks are exactly the shared tracks, with its detections file."""
    for folder in map(Path, dirs):
        provenance_path = folder / f"{recording_id}.provenance.json"
        detections = folder / f"{recording_id}.detections.jsonl"
        if provenance_path.is_file() and detections.is_file():
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            if provenance.get("tracks_sha256") == tracks_sha:
                return folder, detections, provenance
    raise FileNotFoundError(f"No run folder with detections for {recording_id} and tracks {tracks_sha[:12]}")


def _one_to_one(left: list, right: list) -> tuple[int, float, float]:
    """Greedy one-to-one matching by class and IoU >= MATCH_IOU; returns matches and largest differences."""
    free = list(range(len(right)))
    matched, box_diff, score_diff = 0, 0.0, 0.0
    for item in sorted(left, key=lambda d: -d.get("score", 0.0)):
        best = max(((iou(tuple(item["bbox_xyxy"]), tuple(right[i]["bbox_xyxy"])), i) for i in free
                    if right[i]["class_name"] == item["class_name"]), default=(0.0, None))
        if best[1] is not None and best[0] >= MATCH_IOU:
            other = right[best[1]]
            free.remove(best[1])
            matched += 1
            box_diff = max(box_diff, max(abs(a - b) for a, b in zip(item["bbox_xyxy"], other["bbox_xyxy"])))
            if "score" in item:
                score_diff = max(score_diff, abs(item["score"] - other["score"]))
    return matched, box_diff, score_diff


def _compare_native(saved: list, fresh: list) -> dict:
    matched, box_diff, score_diff = _one_to_one(saved, fresh)
    return {"saved_detections": len(saved), "fresh_detections": len(fresh), "matched": matched,
            "max_box_difference_px": round(box_diff, 4), "max_score_difference": round(score_diff, 6),
            "passed": matched == len(saved) == len(fresh)}


def _overlay(scene: dict, data_root: Path, out: Path) -> str:
    from PIL import Image, ImageDraw

    image = Image.open(scenes.scene_image_path(scene, data_root)).convert("RGB")
    draw = ImageDraw.Draw(image)
    width = max(2, image.width // 640)
    for obs in scene["model_input"]["observations"]:
        draw.rectangle(obs["bbox_xyxy"], outline=(255, 64, 0), width=width)
        draw.text((obs["bbox_xyxy"][0] + 3, max(0, obs["bbox_xyxy"][1] - 12 * width)),
                  f"{obs['track_id']} {obs['class_name']}", fill=(255, 255, 0))
    image.thumbnail((1600, 900))
    image.save(out, quality=90)
    return out.name


def main(argv=None) -> int:
    from trackrag.vision import predict_tracks
    from trackrag.vision.__main__ import JsonlSink
    from trackrag.vision.config import load_config
    from trackrag.vision.detector import RFDETRDetector

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--frame-manifest", required=True)
    parser.add_argument("--run-manifest", required=True, help="Hashes of the shared prediction files")
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--profile", action="append", required=True, help="NAME=CONFIG=SAVED_PREDICTIONS_DIR")
    parser.add_argument("--detections-dirs", nargs="+", required=True)
    parser.add_argument("--frames", nargs="+", required=True, help="RECORDING:FRAME")
    parser.add_argument("--fresh", action="append", default=[], help="PROFILE:RECORDING:FRAME")
    parser.add_argument("--out-dir", required=True, help="Overlays and fresh runs (outside Git)")
    parser.add_argument("--record", required=True)
    parser.add_argument("--example", help="PROFILE:RECORDING:FRAME=OUTPUT_JSON")
    args = parser.parse_args(argv)

    data_root, out = Path(args.data_root), Path(args.out_dir)
    targets = [Path(args.record)] + ([Path(args.example.split("=", 1)[1])] if args.example else [])
    for path in [out, *targets]:
        if path.exists():
            raise SystemExit(f"{path} exists; choose a new name or remove the old one first")
    manifest, manifest_sha = scenes.load_frame_manifest(args.frame_manifest)
    run_manifest, run_manifest_sha = scenes.load_run_manifest(args.run_manifest)
    profiles = {}
    for spec in args.profile:
        name, config_path, saved = spec.split("=", 2)
        profiles[name] = (load_config(config_path, weights_dir=args.weights_dir, device=args.device), Path(saved))
    frames = [(item.split(":")[0], int(item.split(":")[1])) for item in args.frames]
    out.mkdir(parents=True)
    record = {"version": 1, "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "frame_manifest_sha256": manifest_sha, "run_manifest_sha256": run_manifest_sha,
              "annotations_folder_present_in_data_root": (data_root / "annotations").exists(),
              "device": args.device,
              "rules": {"detector_match": f"every saved and every fresh raw detection matched one-to-one by class and "
                                          f"IoU >= {MATCH_IOU}",
                        "fresh_scene_match": f"same observations by class and IoU >= {MATCH_IOU}; track numbers may differ"},
              "scenes": [], "fresh_prefix": []}
    kept, detectors = {}, {}
    for name, (config, saved_dir) in profiles.items():
        detectors[name] = RFDETRDetector(config)
        for recording_id, frame_id in frames:
            scene = scenes.load_scene(data_root, recording_id, frame_id, profile=config, predictions_dir=saved_dir,
                                      frame_manifest=manifest, run_manifest=run_manifest)
            scenes.validate_scene(json.loads(json.dumps(scene)), data_root, profile=config, predictions_dir=saved_dir,
                                  frame_manifest=manifest, run_manifest=run_manifest)
            kept[(name, recording_id, frame_id)] = scene
            run_dir, detections, run_provenance = _find_run(args.detections_dirs, recording_id,
                                                            scene["provenance"]["tracks_sha256"])
            saved_row = _detection_row(detections, frame_id)
            one = out / f"oneframe-{name}-{recording_id}-{frame_id}"
            one.mkdir()
            predict_tracks(OneFrame(open_recording(data_root, recording_id), frame_id), config=config,
                           detector=detectors[name], sink=JsonlSink(one, recording_id))
            fresh_row = _detection_row(one / f"{recording_id}.detections.jsonl", frame_id)
            check = _compare_native(saved_row["native"], fresh_row["native"])
            overlay = _overlay(scene, data_root, out / f"overlay-{name}-{recording_id}-{frame_id}.jpg")
            record["scenes"].append({
                "profile": name, "recording_id": recording_id, "frame_id": frame_id,
                "size": [scene["model_input"]["width"], scene["model_input"]["height"]],
                "scene_sha256": scene["scene_sha256"], "image_sha256": scene["model_input"]["image_sha256"],
                "observations": len(scene["model_input"]["observations"]),
                "tracks_sha256": scene["provenance"]["tracks_sha256"],
                "saved_run": {"folder": run_dir.name, "detector_kind": run_provenance.get("detector_kind"),
                              "detections_sha256": hashlib.sha256(detections.read_bytes()).hexdigest()},
                "validated_against_predictions_image_and_manifests": True,
                "detector_one_frame_check": check, "overlay_file": overlay})
            print(f"{name} {recording_id}:{frame_id} observations={len(scene['model_input']['observations'])} "
                  f"detector check {check}", flush=True)
    for spec in args.fresh:
        name, recording_id, frame = spec.split(":")
        config, saved_dir = profiles[name]
        frame_id = int(frame)
        saved = scenes.load_scene(data_root, recording_id, frame_id, profile=config, predictions_dir=saved_dir,
                                  frame_manifest=manifest, run_manifest=run_manifest)
        run_dir = out / f"fresh-{name}-{recording_id}-{frame_id}"
        fresh = scenes.fresh_scene(data_root, recording_id, frame_id, profile=config, out_dir=run_dir,
                                   detector=detectors[name], frame_manifest=manifest)
        saved_obs, fresh_obs = saved["model_input"]["observations"], fresh["model_input"]["observations"]
        matched, box_diff, _ = _one_to_one(saved_obs, fresh_obs)
        run_provenance = json.loads((run_dir / f"{recording_id}.provenance.json").read_text(encoding="utf-8"))
        result = {"profile": name, "recording_id": recording_id, "frame_id": frame_id,
                  "frames_processed": run_provenance["frames"],
                  "origin_labels": [saved["receipt"]["origin"], fresh["receipt"]["origin"]],
                  "run_kinds": [saved["provenance"]["run_kind"], fresh["provenance"]["run_kind"]],
                  "saved_observations": len(saved_obs), "fresh_observations": len(fresh_obs), "matched": matched,
                  "max_box_difference_px": round(box_diff, 4),
                  "same_track_numbers": sorted(o["track_id"] for o in saved_obs) == sorted(o["track_id"] for o in fresh_obs),
                  "same_model_input": saved["model_input"] == fresh["model_input"],
                  "passed": matched == len(saved_obs) == len(fresh_obs)}
        record["fresh_prefix"].append(result)
        print(f"fresh {result}", flush=True)
    record["all_passed"] = (all(s["detector_one_frame_check"]["passed"] for s in record["scenes"])
                            and all(f["passed"] for f in record["fresh_prefix"]))
    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
    Path(args.record).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    if args.example:
        key, target = args.example.split("=", 1)
        name, recording_id, frame = key.split(":")
        Path(target).write_text(json.dumps(kept[(name, recording_id, int(frame))], indent=2) + "\n", encoding="utf-8",
                                newline="\n")
    print(f"all checks passed: {record['all_passed']}")
    return 0 if record["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
