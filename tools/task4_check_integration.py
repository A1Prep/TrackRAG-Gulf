"""Read-only Task 4 artifact preflight; never loads models or performs inference.

Run with explicit artifact paths and at least two --scene RECORDING:FRAME values.
Passing means the checked files/dependency metadata permit a runtime attempt,
not that a model can load on this machine or that its predictions are accurate.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from importlib import metadata, util
import json
from pathlib import Path
import re
import sys


RECORDING_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
STREAMED_MODULES = {"torch": "torch", "torchvision": "torchvision", "transformers": "transformers",
                    "peft": "peft", "accelerate": "accelerate", "bitsandbytes": "bitsandbytes",
                    "safetensors": "safetensors", "huggingface-hub": "huggingface_hub",
                    "tokenizers": "tokenizers", "Pillow": "PIL", "numpy": "numpy"}


def _object(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    def reject(value):
        raise ValueError(f"nonfinite JSON value: {value}")

    with Path(path).open("rb") as source:
        payload = source.read(1024 * 1024 + 1)
    if len(payload) > 1024 * 1024:
        raise ValueError("metadata exceeds the 1 MiB budget")
    result = json.loads(payload.decode("utf-8-sig"), object_pairs_hook=unique, parse_constant=reject)
    if not isinstance(result, dict):
        raise ValueError("metadata must contain an object")
    return result


def _file(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty artifact: {path}")
    # Opening, rather than just stat(), checks read accessibility without loading weights.
    with path.open("rb") as source:
        source.read(1)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size}


def _dependency(distribution, module, expected=None):
    installed = metadata.version(distribution)
    if util.find_spec(module) is None:
        raise ImportError(f"Installed distribution has no importable {module} module")
    if expected is not None and installed != expected:
        raise ValueError(f"{distribution} must be {expected}; installed {installed}")
    return {"distribution": distribution, "version": installed, "module": module,
            "expected_version": expected, "module_imported": False}


def _streamed_pins():
    """Read the same checked-in package pins documented for this runtime."""
    root = Path(__file__).resolve().parents[1]
    pins = {}
    for name in ("requirements-grounding-local.txt", "requirements-grounding-format.txt"):
        for line in (root / name).read_text(encoding="utf-8").splitlines():
            match = re.fullmatch(r"([A-Za-z0-9_-]+)==([^\s#]+)", line.strip())
            if match:
                pins[match[1]] = match[2]
    if not set(STREAMED_MODULES) | {"interegular"} <= set(pins):
        raise ValueError("Streamed runtime requirements are missing an exact supported package pin")
    return pins


def _streamed_configuration(config):
    from trackrag.dropoff.streamed import STREAMED_MODEL_ID, STREAMED_REVISION
    if (config.model_id, config.revision) != (STREAMED_MODEL_ID, STREAMED_REVISION):
        raise ValueError("Streamed runtime requires its separately pinned prequantized 8B distribution")
    if not config.local_files_only:
        raise ValueError("Streamed runtime requires local_files_only=true")
    if config.device == "cpu":
        raise ValueError("Streamed runtime requires CUDA; a CPU device cannot start it")
    if config.quantization != "nf4" or config.dtype not in ("auto", "bfloat16"):
        raise ValueError("Streamed runtime requires NF4 with BF16 compute")
    if not 65536 <= config.min_pixels <= config.max_pixels <= 262144:
        raise ValueError("Streamed image budgets must lie between 65536 and 262144 pixels")
    return {"configuration_compatible": True, "cuda_available": "not_probed", "bf16_supported": "not_probed"}


def _snapshot(config, *, processor=False):
    from trackrag.dropoff.model import BASE_MODEL_ID, BASE_REVISION

    model_id = BASE_MODEL_ID if processor and config.runtime == "streamed" else config.model_id
    revision = BASE_REVISION if processor and config.runtime == "streamed" else config.revision
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*(/[A-Za-z0-9][A-Za-z0-9_.-]*)?", model_id) is None:
        raise ValueError("model_id must be a repository identity, not a filesystem path")
    folder = config.resolved_cache_dir() / ("models--" + model_id.replace("/", "--")) / "snapshots" / revision
    if not folder.is_dir():
        raise FileNotFoundError(f"Pinned local {'processor' if processor else 'base'} snapshot is unavailable: {folder}")
    return folder


def _processor(config):
    folder = _snapshot(config, processor=True)
    files = [_file(folder / "tokenizer_config.json")]
    for alternatives in (("tokenizer.json", "tokenizer.model"),
                         ("preprocessor_config.json", "processor_config.json")):
        selected = next((folder / name for name in alternatives if (folder / name).is_file()), None)
        if selected is None:
            raise FileNotFoundError(f"Missing processor asset, expected one of {alternatives} in {folder}")
        files.append(_file(selected))
    return {"snapshot": str(folder), "files": files, "processor_loaded": False}


def _base(config):
    if config.runtime == "streamed":
        from trackrag.dropoff.streamed import checkpoint_identity

        if config.model_dir is None:
            raise ValueError("streamed runtime requires an explicit model_dir")
        receipt = checkpoint_identity(config.model_dir, verify_weights=False)
        for name in receipt["files"]:
            _file(Path(config.model_dir).expanduser() / name)
        return receipt
    folder = _snapshot(config)
    files = [_file(folder / "config.json")]
    index = folder / "model.safetensors.index.json"
    if index.is_file():
        weight_map = _object(index).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("base weight index must contain a nonempty weight_map")
        shards = set(weight_map.values())
        if any(not isinstance(name, str) or not name or "/" in name or "\\" in name or ":" in name
               or Path(name).name != name for name in shards):
            raise ValueError("base index contains a nonlocal shard name")
        files.append(_file(index))
        files.extend(_file(folder / name) for name in sorted(shards))
    else:
        files.append(_file(folder / "model.safetensors"))
    return {"snapshot": str(folder), "files": files, "verified_weights": False}


def check_integration(*, data_root, vision_config, predictions, grounding_config,
                      scenes, asr_model=None, whisper_model=None, prediction_manifest=None,
                      frame_rate=None, split=None):
    """Retain an independent result for every requested scene and supplied component."""
    checks = []

    def check(identity, function, inputs=None, prerequisite=True):
        row = {"check": identity, "inputs": inputs or {}, "status": "blocked", "details": None, "error": None}
        checks.append(row)
        if not prerequisite:
            row["error"] = {"type": "PrerequisiteError", "message": "Required configuration check failed"}
            return None
        try:
            value = function()
            row.update(status="passed", details=value)
            return value
        except Exception as error:
            row.update(status="error", error={"type": type(error).__name__, "message": str(error)[:4096]})
            return None

    def cohort():
        if len({rid for rid, _ in scenes}) < 2:
            raise ValueError("Select scenes from at least two different recordings")
        return {"requested_scenes": len(scenes), "distinct_recordings": len({rid for rid, _ in scenes})}

    check("recording_cohort", cohort)
    for distribution, module in (("Pillow", "PIL"), ("opencv-python-headless", "cv2")):
        # Either OpenCV distribution is valid for the existing EMT reader.
        def image_dependency(distribution=distribution, module=module):
            if module == "cv2":
                try:
                    return _dependency(distribution, module)
                except metadata.PackageNotFoundError:
                    return _dependency("opencv-python", module)
            return _dependency(distribution, module)
        check(f"dependency:{module}", image_dependency)

    def load_vision():
        from trackrag.vision.config import config_from_dict
        return config_from_dict(_object(vision_config))

    vision = check("vision_configuration", load_vision, {"path": str(vision_config)})
    # Keep dataclass configuration out of the JSON report.
    if vision is not None:
        checks[-1]["details"] = {"name": vision.name, "identity_hash": vision.identity_hash()}
    manifest = check("prediction_manifest", lambda: _object(prediction_manifest),
                     {"path": str(prediction_manifest)}) if prediction_manifest is not None else None
    if prediction_manifest is not None and manifest is not None:
        checks[-1]["details"] = {"path": str(prediction_manifest), "sha256": sha256(Path(prediction_manifest).read_bytes()).hexdigest()}
    for number, (rid, frame_id) in enumerate(scenes, 1):
        def scene(rid=rid, frame_id=frame_id, number=number):
            if not isinstance(rid, str) or RECORDING_NAME.fullmatch(rid) is None:
                raise ValueError("recording_id must be a bounded recording name, not a path")
            if type(frame_id) is not int or frame_id < 0:
                raise ValueError("frame_id must be a nonnegative original frame ID")
            from trackrag.dropoff.app import RecordingPrefix
            from trackrag.dropoff.inputs import build_instruction_request
            from trackrag.emt import open_recording
            from tools.vision_saved_adapters import predict_tracks_with_receipt

            recording = open_recording(data_root, rid, frame_rate=frame_rate, split=split)
            prefix = RecordingPrefix(recording, frame_id)
            rows, receipt = predict_tracks_with_receipt(prefix, directory=predictions, config=vision)
            matched = False
            if prediction_manifest is not None:
                expected = manifest["runs"][vision.identity_hash()][rid]
                if any(receipt.get(key) != expected.get(key) for key in ("tracks_sha256", "provenance_sha256")):
                    raise ValueError("saved prediction bytes differ from the supplied reviewed run manifest")
                matched = True
            request = build_instruction_request(
                recording, rows, frame_id=frame_id, request_id=f"preflight-{number}",
                text="Check the selected scene.", provenance=receipt,
            )
            return {"frame": request["scene"]["frame"], "current_observations": len(request["scene"]["tracks"]),
                    "image_sha256": request["scene"]["provenance"]["image_sha256"],
                    "prediction_receipt": receipt, "reviewed_prediction_manifest_matched": matched,
                    "prediction_image_hash_binding": "not_verified_in_historical_artifact"}
        check(f"scene:{number}", scene, {"recording_id": rid, "frame_id": frame_id},
              prerequisite=vision is not None and (prediction_manifest is None or manifest is not None))

    def load_grounding():
        from trackrag.dropoff.model import GroundingConfig
        return GroundingConfig.from_mapping(_object(grounding_config))

    grounding = check("grounding_configuration", load_grounding, {"path": str(grounding_config)})
    if grounding is not None:
        checks[-1]["details"] = {"model_id": grounding.model_id, "revision": grounding.revision,
                                 "runtime": grounding.runtime, "device": grounding.device,
                                 "local_files_only": grounding.local_files_only}

    def trained_adapter():
        from trackrag.dropoff.model import verify_adapter
        if grounding.adapter_dir is None:
            raise ValueError("A trained-condition check requires adapter_dir; this configuration is pretrained only")
        return verify_adapter(grounding)

    check("trained_adapter", trained_adapter, prerequisite=grounding is not None)
    check("base_checkpoint_accessibility", lambda: _base(grounding), prerequisite=grounding is not None)
    check("processor_accessibility", lambda: _processor(grounding), prerequisite=grounding is not None)
    if grounding is not None and grounding.runtime == "streamed":
        check("streamed_configuration", lambda: _streamed_configuration(grounding))
        pins = check("streamed_requirement_pins", _streamed_pins)
        for distribution, module in STREAMED_MODULES.items():
            check(f"streamed_dependency:{module}",
                  lambda d=distribution, m=module: _dependency(d, m, pins[d]), prerequisite=pins is not None)
        # The release version alone does not establish the Transformers-5 fix.
        from trackrag.dropoff.structured import _dependency_identity
        check("structured_decoder_source", _dependency_identity)
        check("dependency:lmformatenforcer", lambda: _dependency("lm-format-enforcer", "lmformatenforcer"))
        check("dependency:interegular", lambda: _dependency("interegular", "interegular", pins["interegular"]),
              prerequisite=pins is not None)
    else:
        dependencies = [("torch", "torch"), ("transformers", "transformers"), ("peft", "peft"),
                        ("accelerate", "accelerate"), ("safetensors", "safetensors")]
        if grounding is not None and grounding.quantization == "nf4":
            dependencies.append(("bitsandbytes", "bitsandbytes"))
        for distribution, module in dependencies:
            check(f"dependency:{module}", lambda d=distribution, m=module: _dependency(d, m))

    def speech_selection():
        if (asr_model is None) == (whisper_model is None):
            raise ValueError("Supply exactly one native speech selection: --asr-model or --whisper-model")
        return {"backend": "vosk" if asr_model is not None else "whisper"}
    check("speech_selection", speech_selection)
    if asr_model is not None:
        def vosk_files():
            from trackrag.local_speech import MODEL_CONTENT_SHA256, MODEL_ID, _verify_model
            _verify_model(Path(asr_model).expanduser().resolve())
            return {"path": str(Path(asr_model).expanduser().resolve()), "model_id": MODEL_ID,
                    "content_sha256": MODEL_CONTENT_SHA256, "recognizer_loaded": False}
        check("vosk_model", vosk_files, {"path": str(asr_model)})
        check("dependency:vosk", lambda: _dependency("vosk", "vosk", "0.3.45"))
    if whisper_model is not None:
        def whisper_files():
            from trackrag.whisper_speech import MODEL_CONTENT_SHA256, MODEL_ID, REVISION, verify_model
            folder = verify_model(Path(whisper_model).expanduser().resolve())
            return {"path": str(folder), "model_id": MODEL_ID, "revision": REVISION,
                    "content_sha256": MODEL_CONTENT_SHA256, "recognizer_loaded": False}
        check("whisper_model", whisper_files, {"path": str(whisper_model)})
        from trackrag.whisper_speech import VERSIONS
        for distribution, module in (("faster-whisper", "faster_whisper"), ("ctranslate2", "ctranslate2"), ("av", "av")):
            check(f"dependency:{module}", lambda d=distribution, m=module: _dependency(d, m, VERSIONS[d]))
    ready = all(row["status"] == "passed" for row in checks)
    return {"schema_version": 1, "kind": "task4_real_module_preflight", "status": "ready_for_runtime_attempt" if ready else "blocked",
            "ready_for_runtime_attempt": ready, "checks": checks,
            "execution": {"development_adapters": None, "fresh_detector_inference": False,
                          "grounding_inference": False, "speech_recognition": False, "models_loaded": False,
                          "downloads": False, "files_written": False},
            "limits": ["File and dependency checks do not establish model load, GPU/ABI compatibility, inference quality or end-to-end acceptance.",
                       "Matching adapter manifest hashes establish supplied file consistency, not independent training provenance or improvement.",
                       "Base weight files are checked for accessibility, not fully hashed; the production loader retains its checks.",
                       "Saved predictions without a reviewed manifest match have verified consistency, not independently established production provenance.",
                       "Historical prediction bundles do not bind individual image bytes."]}


def _scene(value):
    try:
        recording, original_frame = value.rsplit(":", 1)
        frame = int(original_frame)
        if RECORDING_NAME.fullmatch(recording) is None or frame < 0:
            raise ValueError
    except (ValueError, AttributeError) as error:
        raise argparse.ArgumentTypeError("scene must be RECORDING:NONNEGATIVE_FRAME_ID") from error
    return recording, frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "vision-config", "predictions", "grounding-config"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--scene", type=_scene, action="append", required=True)
    parser.add_argument("--prediction-manifest", type=Path, help="Optional reviewed identity/recording hash manifest")
    parser.add_argument("--asr-model", type=Path, help="Pinned Vosk model folder; does not initialize recognition")
    parser.add_argument("--whisper-model", type=Path, help="Pinned Whisper Small folder; does not initialize recognition")
    parser.add_argument("--frame-rate", type=float)
    parser.add_argument("--split", choices=("train", "test"))
    args = vars(parser.parse_args(argv))
    args["scenes"] = args.pop("scene")
    report = check_integration(**args)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["ready_for_runtime_attempt"] else 1


if __name__ == "__main__":
    sys.exit(main())
