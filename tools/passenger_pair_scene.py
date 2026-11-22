"""Prepare the optional frame-781 passenger bookmark from the private Task 2 photo and Task 1 predictions."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from tools.dropoff_import_task2_review import PACKAGE, REVISION, git_blob
from tools.vision_saved_adapters import load_track_artifact
from trackrag.dropoff.contracts import validate_request
from trackrag.dropoff.data import file_sha256
from trackrag.dropoff.inputs import _inspect_image
from trackrag.dropoff.scenes import load_frame_manifest, load_run_manifest
from trackrag.emt import DEFAULT_FRAME_RATE
from trackrag.vision.config import load_config


ROOT = Path(__file__).resolve().parents[1]
RECORDING = "video_054907"
FRAME_ID = 781
PHOTO_IN_TASK2 = "images/development/video_054907/00781.jpg"


def _files(predictions, temporary):
    names = [f"{RECORDING}.tracks.jsonl", f"{RECORDING}.provenance.json"]
    if predictions.is_dir():
        return [predictions / name for name in names]
    result = []
    with ZipFile(predictions) as archive:
        for name in names:
            selected = [
                entry for entry in archive.infolist()
                if entry.filename.replace("\\", "/").split("/")[-1] == name
            ]
            if len(selected) != 1 or selected[0].file_size > 64 * 1024**2:
                raise ValueError(f"The supplied prediction ZIP must contain one bounded {name}")
            dest = temporary / name
            with archive.open(selected[0]) as source, dest.open("xb") as target:
                shutil.copyfileobj(source, target)
            result.append(dest)
    return result


def prepare(predictions, out, *, image=None, repo=ROOT):
    out = Path(out).resolve()
    image_registry, frame_registry_hash = load_frame_manifest(ROOT / "experiments/vision/emt_frame_sha256_manifest.json")
    run_registry, run_registry_hash = load_run_manifest(ROOT / "experiments/vision/shared_prediction_runs.json")
    identity = load_config(ROOT / "configs/vision/finetuned.json").identity_hash()
    expected = run_registry["runs"][identity][RECORDING]
    image_hash = image_registry["recordings"][RECORDING][str(FRAME_ID)]
    photo = Path(image).read_bytes() if image is not None else git_blob(Path(repo), PHOTO_IN_TASK2)
    if sha256(photo).hexdigest() != image_hash:
        raise ValueError("The JPEG differs from the registered original EMT frame")
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent, prefix=".pair-scene-") as temp:
        tmp = Path(temp)
        tracked, provenance = _files(Path(predictions), tmp)
        source_bytes = provenance.read_bytes()
        if sha256(source_bytes).hexdigest() != expected["provenance_sha256"]:
            raise ValueError("Different Task 1 saved run provenance")
        raw = json.loads(source_bytes)
        if raw["identity_hash"] != identity or raw["tracks_sha256"] != expected["tracks_sha256"]:
            raise ValueError("Task 1 fine-tuned detector identity or track-file digest differs")
        rows, frames, size, tracks_hash = load_track_artifact(tracked, raw, RECORDING)
        if FRAME_ID not in frames or tracks_hash != expected["tracks_sha256"]:
            raise ValueError("The registered Task 1 artifact does not supply this frame")
        current = [dict(r, bbox_xyxy=list(r["bbox_xyxy"])) for r in rows if r["frame_id"] == FRAME_ID]
        stage = tmp / "private-scene"
        (stage / "images").mkdir(parents=True)
        rel = Path("images") / "00781.jpg"
        (stage / rel).write_bytes(photo)
        frame = {
            "recording_id": RECORDING, "frame_id": FRAME_ID,
            "timestamp_s": (FRAME_ID - frames[0]) / DEFAULT_FRAME_RATE,
            "width": size[0], "height": size[1], "image_path": str(stage / rel),
        }
        _, decoded_source_hash = _inspect_image(frame)
        if decoded_source_hash != image_hash:
            raise ValueError("Original photo changed during reconstruction")
        frame["image_path"] = rel.as_posix()
        scene = {
            "frame": frame, "source": "predicted", "tracks": current,
            "provenance": {
                "image_sha256": image_hash, "tracks_sha256": tracks_hash,
                "prediction_provenance_sha256": expected["provenance_sha256"],
                "vision_identity_hash": identity, "vision_source": "validated_saved_predictions",
            },
        }
        row = validate_request({
            "request_id": "open-optional-pair-scene", "input_mode": "typed",
            "dialogue": [], "text": "Open this recorded scene.", "scene": scene,
        })
        bank = stage / "requests.jsonl"
        bank.write_text(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf8", newline="\n")
        receipt = {
            "purpose": "one additional local passenger bookmark", "recording": RECORDING, "frame": FRAME_ID,
            "original_photo_sha256": image_hash, "frame_registry_sha256": frame_registry_hash,
            "prediction_registry_sha256": run_registry_hash, "tracks_sha256": tracks_hash,
            "prediction_provenance_sha256": expected["provenance_sha256"], "detector_identity_sha256": identity,
            "task2_revision": REVISION, "task2_package": PACKAGE, "original_photo_path_in_task2": PHOTO_IN_TASK2,
            "frame_rate": DEFAULT_FRAME_RATE, "timestamp_s": frame["timestamp_s"],
            "current_predicted_observations": len(current), "saved_predictions": True, "new_detector_run": False,
            "human_target_rows_loaded": 0, "instruction_examples_for_server": 0, "footage_prefix_expanded": False,
            "relative_image": rel.as_posix(), "request_bank_sha256": file_sha256(bank),
        }
        (stage / "source.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf8", newline="\n")
        if out.exists():
            for member in [Path("requests.jsonl"), Path("source.json"), rel]:
                original = out / member
                if not original.is_file() or file_sha256(original) != file_sha256(stage / member):
                    raise ValueError(f"Existing {original} differs. Supply a fresh output directory.")
        else:
            stage.rename(out)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="Task 1 fine-tuned prediction ZIP or its extracted prediction directory")
    parser.add_argument("--image", type=Path,
                        help="An original licensed JPEG. By default the importer reads the pinned Task 2 Git blob.")
    parser.add_argument("--repo", type=Path, default=ROOT, help="Private Git checkout with the Task 2 handoff commit")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/passenger-pair-scene")
    args = parser.parse_args()
    print(json.dumps(prepare(args.predictions, args.out, image=args.image, repo=args.repo), indent=2))


if __name__ == "__main__":
    main()
