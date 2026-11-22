"""Prepare annotation-free grounding requests from real EMT saved predictions.

This connects existing Task 1/2 artifacts to Task 5. It does not call a language
model, invent destinations, or replace Task 1's independent evaluation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

from trackrag.dropoff.contracts import validate_request
from trackrag.dropoff.inputs import build_instruction_request
from trackrag.emt import open_recording
from trackrag.vision.config import config_from_dict
from tools.vision_saved_adapters import predict_tracks_with_receipt

MAX_INPUT_BYTES = 16 * 1024 * 1024
MAX_ROWS = 10_000
SPEC_REQUIRED = frozenset({"request_id", "recording_id", "frame_id", "text"})
SPEC_OPTIONAL = frozenset({"input_mode", "dialogue", "original_transcript"})
RECORDING_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON field: {key}")
        value[key] = item
    return value


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def _parse_json(text):
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def read_specifications(payload):
    """Reject ambiguous bank identities; retain identifiable bad rows for receipts."""
    if not isinstance(payload, bytes) or len(payload) > MAX_INPUT_BYTES:
        raise ValueError("instruction bank exceeds the 16 MiB byte limit")
    result, seen = [], set()
    for number, line in enumerate(payload.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        if len(result) >= MAX_ROWS or len(line) > 2_000_000:
            raise ValueError("instruction bank exceeds bounded row limits")
        try:
            row = _parse_json(line)
        except ValueError as error:
            raise ValueError(f"instruction line {number}: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"instruction line {number} must be an object")
        identity = row.get("request_id")
        if not isinstance(identity, str) or not identity.strip() or len(identity) > 200:
            raise ValueError(f"instruction line {number} needs a nonempty request_id")
        if identity in seen:
            raise ValueError(f"duplicate request_id: {identity}")
        seen.add(identity)
        result.append(row)
    if not result:
        raise ValueError("instruction bank is empty")
    return result


def _validate_specification(row):
    missing = SPEC_REQUIRED - row.keys()
    extra = row.keys() - SPEC_REQUIRED - SPEC_OPTIONAL
    if missing or extra:
        raise ValueError(f"instruction has missing {sorted(missing)} or unknown {sorted(extra)} fields")
    rid = row["recording_id"]
    if not isinstance(rid, str) or RECORDING_NAME.fullmatch(rid) is None:
        raise ValueError("recording_id must be a bounded recording name, not a path")
    if type(row["frame_id"]) is not int or row["frame_id"] < 0:
        raise ValueError("frame_id must be an original nonnegative integer")


class _RecordingPrefix:
    """A true available-frame prefix; never substitute the selected frame alone."""

    def __init__(self, recording, cutoff):
        self._recording = recording
        self.recording_id = recording.recording_id
        self.frame_ids = [fid for fid in recording.frame_ids if fid <= cutoff]
        self.cutoff = cutoff
        if cutoff not in self.frame_ids:
            raise KeyError(f"Frame {cutoff} is absent from recording {self.recording_id!r}")

    def iter_frames(self):
        for fid in self.frame_ids:
            yield self.get_frame(fid)

    def get_frame(self, frame_id):
        if frame_id > self.cutoff or frame_id not in self.frame_ids:
            raise KeyError("frame lies outside the requested recording prefix")
        return self._recording.get_frame(frame_id)


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_scene_requests(instruction_snapshot, *, data_root, predictions_dir,
                           vision_config, output_dir, frame_rate=None, split=None):
    specifications = read_specifications(instruction_snapshot)
    if frame_rate is not None and (type(frame_rate) not in (int, float)
                                   or not math.isfinite(frame_rate) or frame_rate <= 0):
        raise ValueError("frame_rate must be finite and positive")
    data_root = Path(data_root).resolve()
    predictions_dir = Path(predictions_dir).resolve()
    config_path = Path(vision_config).resolve()
    if config_path.stat().st_size > 1024 * 1024:
        raise ValueError("vision configuration exceeds 1 MiB")
    config_snapshot = config_path.read_bytes()
    config = config_from_dict(_parse_json(config_snapshot.decode("utf-8-sig")),
                              source_path=str(config_path))
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "instructions.snapshot.jsonl").write_bytes(instruction_snapshot)
    (output_dir / "vision-config.snapshot.json").write_bytes(config_snapshot)
    images = output_dir / "images"
    images.mkdir()
    run = {
        "schema_version": 1, "kind": "emt_predicted_grounding_request_preparation",
        "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
        "instructions_sha256": hashlib.sha256(instruction_snapshot).hexdigest(),
        "vision_config_sha256": hashlib.sha256(config_snapshot).hexdigest(),
        "vision_identity_hash": config.identity_hash(),
        "data_root": str(data_root), "saved_predictions": str(predictions_dir),
        "frame_rate_override": frame_rate, "split": split,
        "requested_rows": len(specifications), "prepared_rows": 0, "preparation_errors": 0,
        "request_file": "requests.jsonl", "outcome_file": "preparation-outcomes.jsonl",
        "prediction_receipts": "prediction-receipts.jsonl",
        "track_source": "predicted", "observation_policy": "selected_frame_only",
        "annotation_labels_used": False, "fresh_detector_inference": False,
        "language_model_called": False, "grounding_quality_measured": False,
    }
    _write_json(output_dir / "run.json", run)
    recordings, saved = {}, {}
    with (output_dir / "requests.jsonl").open("w", encoding="utf-8", newline="\n") as requests_file, \
         (output_dir / "preparation-outcomes.jsonl").open("w", encoding="utf-8", newline="\n") as outcomes_file, \
         (output_dir / "prediction-receipts.jsonl").open("w", encoding="utf-8", newline="\n") as receipts_file:
        for row in specifications:
            identity = row["request_id"]
            try:
                _validate_specification(row)
                rid, cutoff = row["recording_id"], row["frame_id"]
                if rid not in recordings:
                    recordings[rid] = open_recording(data_root, rid, frame_rate=frame_rate, split=split)
                recording = recordings[rid]
                key = (rid, cutoff)
                if key not in saved:
                    bounded = _RecordingPrefix(recording, cutoff)
                    observations, receipt = predict_tracks_with_receipt(
                        bounded, directory=predictions_dir, config=config,
                    )
                    receipt["recording_frame_rate"] = recording.frame_rate
                    saved[key] = (observations, receipt)
                    receipts_file.write(json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n")
                    receipts_file.flush()
                observations, receipt = saved[key]
                request = build_instruction_request(
                    recording, observations, frame_id=cutoff, request_id=identity, text=row["text"],
                    input_mode=row.get("input_mode", "typed"), dialogue=row.get("dialogue"),
                    original_transcript=row.get("original_transcript"),
                    provenance={
                        "vision_source": "validated_saved_predictions",
                        "vision_identity_hash": receipt["vision_identity_hash"],
                        "tracks_sha256": receipt["tracks_sha256"],
                        "prediction_provenance_sha256": receipt["provenance_sha256"],
                        "prediction_image_hash_binding": "not_verified_in_historical_artifact",
                    },
                )
                original_path = Path(request["scene"]["frame"]["image_path"])
                with original_path.open("rb") as image_file:
                    image_bytes = image_file.read(32 * 1024 * 1024 + 1)
                if len(image_bytes) > 32 * 1024 * 1024:
                    raise ValueError("original image exceeds the byte limit")
                image_hash = hashlib.sha256(image_bytes).hexdigest()
                if image_hash != request["scene"]["provenance"]["image_sha256"]:
                    raise ValueError("original image changed after scene validation")
                suffix = original_path.suffix.lower()
                if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    suffix = ".image"
                relative_image = Path("images") / (image_hash + suffix)
                staged = output_dir / relative_image
                if staged.exists():
                    if hashlib.sha256(staged.read_bytes()).hexdigest() != image_hash:
                        raise ValueError("existing image snapshot hash does not match")
                else:
                    with staged.open("xb") as image_file:
                        image_file.write(image_bytes)
                request["scene"]["frame"]["image_path"] = relative_image.as_posix()
                request = validate_request(request)
                encoded = json.dumps(request, sort_keys=True, allow_nan=False)
                requests_file.write(encoded + "\n")
                requests_file.flush()
                run["prepared_rows"] += 1
                outcome = {
                    "ok": True, "request_id": identity, "recording_id": rid, "frame_id": cutoff,
                    "request_line": run["prepared_rows"], "image_sha256": image_hash,
                    "request_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
                    "current_frame_observations": len(request["scene"]["tracks"]),
                }
            except (ValueError, OSError, KeyError, TypeError) as error:
                run["preparation_errors"] += 1
                outcome = {
                    "ok": False, "request_id": identity,
                    "error": {"code": "scene_preparation_failed", "type": type(error).__name__,
                              "message": str(error)[:2000]},
                }
            outcomes_file.write(json.dumps(outcome, sort_keys=True, allow_nan=False) + "\n")
            outcomes_file.flush()
    run["status"] = "completed_with_errors" if run["preparation_errors"] else "complete"
    run["completed_at"] = datetime.now(timezone.utc).isoformat()
    for filename in ("requests.jsonl", "preparation-outcomes.jsonl", "prediction-receipts.jsonl"):
        run[filename.removesuffix(".jsonl").replace("-", "_") + "_sha256"] = hashlib.sha256(
            (output_dir / filename).read_bytes()
        ).hexdigest()
    _write_json(output_dir / "run.json", run)
    return run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="Task 1 saved .tracks.jsonl/.provenance.json folder")
    parser.add_argument("--vision-config", type=Path, required=True,
                        help="Explicit configuration whose identity produced those predictions")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--instructions", type=Path,
                        help="JSONL instruction specifications, including multiple recordings")
    parser.add_argument("--recording")
    parser.add_argument("--frame", type=int)
    parser.add_argument("--text")
    parser.add_argument("--request-id")
    parser.add_argument("--input-mode", choices=("typed", "voice"), default="typed")
    parser.add_argument("--original-transcript")
    parser.add_argument("--frame-rate", type=float)
    parser.add_argument("--split", choices=("train", "test"))
    args = parser.parse_args(argv)
    single = (args.recording, args.frame, args.text, args.request_id)
    if args.instructions:
        if any(value is not None for value in single) or args.original_transcript is not None or args.input_mode != "typed":
            parser.error("--instructions cannot be combined with single-request fields")
    elif any(value is None for value in single):
        parser.error("supply --instructions or all of --recording, --frame, --text and --request-id")
    try:
        if args.instructions:
            if args.instructions.stat().st_size > MAX_INPUT_BYTES:
                raise ValueError("instruction bank exceeds the byte limit")
            snapshot = args.instructions.read_bytes()
        else:
            row = {"request_id": args.request_id, "recording_id": args.recording,
                   "frame_id": args.frame, "text": args.text, "input_mode": args.input_mode}
            if args.original_transcript is not None:
                row["original_transcript"] = args.original_transcript
            snapshot = (json.dumps(row, allow_nan=False) + "\n").encode("utf-8")
        result = prepare_scene_requests(
            snapshot, data_root=args.data_root, predictions_dir=args.predictions,
            vision_config=args.vision_config, output_dir=args.out,
            frame_rate=args.frame_rate, split=args.split,
        )
        print(json.dumps(result, indent=2))
        return 1 if result["preparation_errors"] else 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"Scene preparation failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
