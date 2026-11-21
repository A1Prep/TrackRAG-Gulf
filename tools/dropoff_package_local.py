"""Stage verified local public-destination artifacts without modifying a raw run.

This records completed training and actual reload/probe evidence. It neither runs
models nor turns public geometry supervision into local clarification training.
Use dropoff_bundle artifacts/verify on the newly created staging directory.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re
import stat
import subprocess
import tempfile
import zipfile

from tools.dropoff_bundle import (ADAPTER_FILES, ARTIFACT_LIMIT, BundleError,
                                 _json, _regular_bytes, artifact_allowed, source_allowed)
from tools.dropoff_compare_local import compare
from trackrag.dropoff.contracts import validate_request
from trackrag.dropoff.formatting import parse_model_output
from trackrag.dropoff.model import BASE_MODEL_ID, BASE_REVISION, GroundingConfig
from trackrag.dropoff.streamed import (METADATA_SHA256, STREAMED_MODEL_ID, STREAMED_REVISION,
                                      WEIGHT_SHA256)


SOURCE_FILES = {
    "tools/dropoff_fit_local.py", "tools/dropoff_train.py", "trackrag/dropoff/model.py",
    "trackrag/dropoff/streamed.py", "trackrag/dropoff/streamed_weights.py",
    "trackrag/dropoff/streamed_edges.py", "trackrag/dropoff/formatting.py",
    "trackrag/dropoff/contracts.py", "trackrag/dropoff/data.py",
}
PROBE_FILES = {
    "run.json": "production_probe_summary.json",
    "model-outcomes.jsonl": "production_probe_outcomes.jsonl",
    "receipts.jsonl": "production_probe_receipts.jsonl",
    "resolved-requests.jsonl": "production_probe_requests.jsonl",
    "input-snapshot.jsonl": "production_probe_input_snapshot.jsonl",
}
EVALUATION_FILES = {
    "pretrained.jsonl", "pretrained.summary.json", "pretrained.teacher-metrics.json",
    "trained.jsonl", "trained.summary.json", "trained.teacher-metrics.json",
    "trained.adapter-hashes.json", "comparison.json",
    "contrast-pretrained.jsonl", "contrast-pretrained.summary.json",
    "contrast-trained.jsonl", "contrast-trained.summary.json",
    "contrast-trained.adapter-hashes.json", "contrast-comparison.json",
}
OPTIONAL_RUN_FILES = {
    "evaluation_cohort.json", "runtime_provenance.json", "hardware.json",
    "environment_freeze.txt", "training-log.jsonl", "pretrained_teacher_metrics.json",
    "epoch-1-teacher-metrics.json", "epoch-2-teacher-metrics.json", "software-validation.json",
}
_SHA = re.compile(r"[0-9a-f]{64}")


class PackageError(BundleError):
    pass


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def require(condition, message):
    if not condition:
        raise PackageError(message)


def hash_map(value, label):
    require(isinstance(value, dict) and bool(value) and all(isinstance(k, str) and isinstance(v, str)
            and _SHA.fullmatch(v) for k, v in value.items()), f"Invalid {label} tensor hashes")
    return value


def inference_profile(value, selected, label):
    config = GroundingConfig.from_mapping(value)
    require(config.runtime == "streamed" and config.quantization == "nf4"
            and config.dtype in ("auto", "bfloat16") and config.device != "cpu"
            and config.local_files_only is True and bool(config.model_dir),
            label + " is not the supported local streamed NF4/BF16 condition")
    require((config.model_id, config.revision) == (selected["base_model_id"], selected["base_revision"])
            and config.suffix_layers == selected["suffix_layers"]
            and all(getattr(config, key) == selected["image_processing"][key]
                    for key in ("min_pixels", "max_pixels")), label + " differs from the selected checkpoint profile")
    return config


def committed_protocol(path, blob):
    """The supplied frozen protocol must match a file in the current Git HEAD."""
    path = Path(path).resolve()
    try:
        root = Path(subprocess.run(["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
            check=True, capture_output=True, text=True).stdout.strip()).resolve()
        relative = path.relative_to(root).as_posix()
        committed = subprocess.run(["git", "-C", str(root), "show", "HEAD:" + relative],
                                   check=True, capture_output=True).stdout
        revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise PackageError("Frozen protocol must be a committed file in the checkout") from error
    require(committed == blob, "Frozen protocol differs from its committed bytes")
    return {"commit": revision, "path": relative, "sha256": digest(blob)}


def source_archive(run, settings):
    expected = settings.get("source_sha256")
    require(isinstance(expected, dict) and set(expected) == SOURCE_FILES,
            "Training settings must bind exactly the nine captured source files")
    output = BytesIO()
    files = {}
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(SOURCE_FILES):
            require(source_allowed(name), "Captured source is outside the source allowlist")
            _, blob = _regular_bytes(run / "source" / name, run, max_bytes=8 * 1024**2)
            require(digest(blob) == expected[name], "Captured training source hash differs: " + name)
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, blob)
            files[name] = {"sha256": digest(blob), "bytes": len(blob)}
    blob = output.getvalue()
    return blob, {"schema_version": 1, "kind": "exact_captured_training_source_overlay",
                  "files": files, "archive_sha256": digest(blob),
                  "purpose": "training_reproduction_only_in_separate_copy",
                  "production_overlay_allowed": False,
                  "note": "Use these exact nine files only in a separate training-reproduction copy. "
                          "Use the current source bundle for production inference; overlaying the old training "
                          "snapshot there would remove later interpretation and decoding repairs."}


def validate_comparisons(payloads, protocol_blob, dataset):
    """Run the existing comparison against the exact bytes that will be staged."""
    reports = {}
    with tempfile.TemporaryDirectory(prefix="trackrag-package-check-") as directory:
        root = Path(directory)
        for name, blob in payloads.items():
            if name.startswith("evaluation/"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(blob)
        frozen = root / "protocol.json"
        frozen.write_bytes(protocol_blob)
        for prefix, report_name, protocol in (("", "comparison.json", None),
                                             ("contrast-", "contrast-comparison.json", frozen)):
            computed = compare(root / f"evaluation/{prefix}pretrained.jsonl",
                               root / f"evaluation/{prefix}trained.jsonl", protocol=protocol)
            saved = _json(payloads["evaluation/" + report_name])
            require(saved == computed, "Saved comparison differs from recomputed evidence: " + report_name)
            require(computed["development_source_sha256"] == dataset["development_sha256"],
                    "Comparison references a different development dataset")
            require(set(computed["case_ids"]) <= set(dataset["development_case_ids"]),
                    "Comparison contains cases outside the declared development partition")
            reports[report_name] = computed
    return reports


def validate_production_probe(probe, probe_image, production, selected, selected_sha, dataset):
    from PIL import Image
    require(probe.get("schema_version") == 1 and probe.get("partition") == "train",
            "Reload probe must be an explicitly public training probe")
    require(probe.get("case_id") in dataset["train_case_ids"], "Probe is not a declared training case")
    portable = validate_request(probe["request"])
    frame = portable["scene"]["frame"]
    require(frame["image_path"] in ("probe-image.jpg", "probe-image.png", "probe-image.webp"),
            "Probe image name is not portable")
    require(portable["scene"].get("provenance", {}).get("mode") == "image_only_public_training",
            "Only the approved public image-only training probe can be packaged")
    image_sha = digest(probe_image)
    require(probe.get("image_sha256") == image_sha, "Portable probe image hash differs")
    artifacts = {row["case_id"]: row for row in dataset["case_artifacts"]}
    require(artifacts[probe["case_id"]]["image_sha256"] == image_sha,
            "Public probe image differs from its training-manifest binding")
    with Image.open(BytesIO(probe_image)) as image:
        require(image.size == (frame["width"], frame["height"]), "Probe dimensions differ")

    summary = _json(production["run.json"])
    configuration = summary.get("configuration", {})
    config = inference_profile(configuration, selected, "Production configuration")
    require(bool(config.adapter_dir), "Production probe configuration has no trained adapter")
    bindings = (("requests_sha256", "input-snapshot.jsonl"),
                ("resolved_requests_sha256", "resolved-requests.jsonl"),
                ("outcomes_sha256", "model-outcomes.jsonl"), ("receipts_sha256", "receipts.jsonl"))
    for field, filename in bindings:
        require(summary.get(field) == digest(production[filename]), "Production probe file binding differs: " + filename)
    records = {}
    for name in ("resolved-requests.jsonl", "model-outcomes.jsonl", "receipts.jsonl"):
        rows = [_json(line.encode()) for line in production[name].decode("utf-8").splitlines() if line.strip()]
        require(len(rows) == 1, "Production probe must contain exactly one row: " + name)
        records[name] = rows[0]
    actual = validate_request(records["resolved-requests.jsonl"])
    normalized = deepcopy(actual)
    normalized["scene"]["frame"]["image_path"] = frame["image_path"]
    require(normalized == portable, "Production request differs from the portable probe")
    receipt, outcome = records["receipts.jsonl"], records["model-outcomes.jsonl"]
    provenance = receipt.get("provenance", {})
    adapter = provenance.get("adapter", {})
    require(summary.get("requested_rows") == 1 and summary.get("semantic_results") == 1
            and summary.get("operational_errors") == 0 and outcome.get("ok") is True
            and receipt.get("status") == "completed", "Production probe did not complete successfully")
    require(receipt.get("request") == actual and outcome.get("request_id") == actual["request_id"]
            and receipt.get("request_id") == actual["request_id"], "Production request/receipt identity differs")
    require(provenance.get("backend_kind") == "transformers_local_streamed"
            and provenance.get("injected") is False and provenance.get("requested_condition") == "fine_tuned",
            "Production probe is not a genuine fine-tuned streamed condition")
    require((provenance.get("model_id"), provenance.get("revision")) ==
            (selected["base_model_id"], selected["base_revision"]), "Production base identity differs")
    require(provenance.get("request_sha256") == digest(canonical(actual))
            and provenance.get("image", {}).get("sha256") == image_sha,
            "Production request/image hash differs")
    require(adapter.get("manifest_sha256") == selected_sha, "Production probe used a different selected manifest")
    require((adapter.get("base_model_id"), adapter.get("base_revision")) ==
            (selected["base_model_id"], selected["base_revision"]), "Production adapter base differs")
    require({name: row.get("sha256") for name, row in adapter.get("files", {}).items()} == selected["files"],
            "Production probe adapter file hashes differ")
    require(outcome.get("provenance") == provenance, "Production outcome/receipt provenance differs")
    details = provenance.get("backend_details", {})
    identity = details.get("weight_identity", {})
    require(details.get("native_fault_isolation") == "spawn_process"
            and type(details.get("worker_pid")) is int and details["worker_pid"] > 0
            and details.get("runtime") == "streamed" and details.get("dtype") == "bfloat16"
            and str(details.get("device", "")).startswith("cuda")
            and (details.get("model_id"), details.get("revision")) == (STREAMED_MODEL_ID, STREAMED_REVISION)
            and (details.get("processor_model_id"), details.get("processor_revision")) == (BASE_MODEL_ID, BASE_REVISION)
            and details.get("suffix_layers") == config.suffix_layers
            and details.get("effective_resident_layers") == config.inference_resident_layers
            and details.get("image_processing") == selected["image_processing"],
            "Production receipt lacks the genuine native worker/profile details")
    require(identity.get("verified_weights") is True and identity.get("verified_metadata") is True
            and (identity.get("model_id"), identity.get("revision")) == (STREAMED_MODEL_ID, STREAMED_REVISION),
            "Production base weight identity was not verified")
    require(all(identity.get("files", {}).get(name, {}).get("sha256") == expected
                for name, expected in {**WEIGHT_SHA256, **METADATA_SHA256}.items()),
            "Production base file digests differ from the pinned checkpoint")
    parsed = parse_model_output(receipt["raw_text"], frame["width"], frame["height"])
    require(parsed.get("status") == "proposed" and parsed == outcome.get("result") == receipt.get("result"),
            "Production probe lacks a full, consistently parsed spatial proposal")
    return {"request_sha256": provenance["request_sha256"], "image_sha256": image_sha,
            "request_id": actual["request_id"], "result": parsed,
            "original_selected_manifest_sha256": selected_sha,
            "production_summary_sha256": digest(production["run.json"]),
            "production_receipts_sha256": digest(production["receipts.jsonl"]),
            "note": "Public training-case reload smoke check, not quality or final EMT evaluation."}


def stage_run(run_dir, production_probe_run, probe_input, protocol, out):
    run, production_root, probe_root, out = (Path(path).resolve() for path in
        (run_dir, production_probe_run, probe_input, out))
    require(not out.exists(), "Choose a new staging directory")
    require(not any(out.is_relative_to(root) for root in (run, production_root, probe_root)),
            "Staging must be separate from every raw input directory")
    payloads = {}
    def read(name):
        if name not in payloads:
            _, blob = _regular_bytes(run / name, run, max_bytes=ARTIFACT_LIMIT - sum(map(len, payloads.values())))
            payloads[name] = blob
        return payloads[name]
    settings, dataset, raw = (_json(read(name)) for name in
        ("training_settings.json", "dataset_manifest.json", "training_result.json"))
    selected_blob = read("adapter/grounding_adapter_manifest.json")
    selected = _json(selected_blob)
    require(raw.get("status") == "public_destination_finetuning_completed", "Full training has not completed")
    require(settings.get("stage") == "public_destination_geometry", "Unsupported training stage")
    model_config = settings["model_configuration"]
    require((selected.get("base_model_id"), selected.get("base_revision")) ==
            (STREAMED_MODEL_ID, STREAMED_REVISION) == (model_config.get("model_id"), model_config.get("revision")),
            "Selected checkpoint and training base identities differ")
    n, epochs, accumulation = len(dataset["train_case_ids"]), settings["epochs"], settings["gradient_accumulation"]
    require(n > 0 and len(set(dataset["train_case_ids"])) == n and type(epochs) is int and epochs > 0
            and type(accumulation) is int and accumulation > 0, "Invalid training schedule")
    updates, presentations = math.ceil(n / accumulation) * epochs, n * epochs
    require(raw.get("optimizer_steps") == updates and raw.get("examples_consumed") == presentations
            and raw.get("epochs_completed") == epochs and raw.get("epochs") == epochs
            and raw.get("training_examples_unique") == n, "Completed full schedule counts differ")
    from collections import Counter
    require(Counter(raw.get("actual_training_case_ids", [])) == Counter({key: epochs for key in dataset["train_case_ids"]}),
            "Actual training presentations do not cover the full declared schedule")
    epoch = raw["selected_checkpoint"]["epoch"]
    require(type(epoch) is int and 1 <= epoch <= epochs, "Invalid selected checkpoint epoch")
    require(selected.get("optimizer_steps") == math.ceil(n / accumulation) * epoch
            and selected.get("actual_training_examples_consumed") == n * epoch,
            "Selected checkpoint counts differ from its selected epoch")
    initial = hash_map(raw["adapter_hashes_initial"], "initial")
    selected_hashes = hash_map(selected["adapter_parameter_hashes"], "selected")
    require(set(initial) == set(selected_hashes), "Initial/selected tensor identities differ")
    require(raw.get("selected_adapter_parameter_hashes") == selected_hashes,
            "Raw training result selected tensor hashes differ")
    changed = sum(initial[key] != selected_hashes[key] for key in initial)
    require(changed > 0, "Selected adapter contains no changed learned tensors")
    expected_files = selected["files"]
    require(set(expected_files) == {"adapter_config.json", "adapter_model.safetensors"}, "Invalid adapter file binding")
    require(raw.get("selected_adapter_files") == expected_files, "Raw result selected adapter file hashes differ")
    for name, expected in expected_files.items():
        require(digest(read("adapter/" + name)) == expected, "Selected adapter file changed: " + name)
    require(selected.get("training_manifest_sha256") == digest(payloads["dataset_manifest.json"])
            and selected.get("training_settings_sha256") == digest(payloads["training_settings.json"]),
            "Selected adapter training-data/settings bindings differ")
    require(not set(dataset["train_scene_groups"]) & set(dataset["development_scene_groups"]),
            "Training and development scenes overlap")
    for name in EVALUATION_FILES:
        path = run / "evaluation" / name
        if path.exists(): read("evaluation/" + name)
    for name in ("trained.adapter-hashes.json", "contrast-trained.adapter-hashes.json"):
        require(_json(read("evaluation/" + name)) == selected_hashes,
                "Fresh evaluation tensor hashes differ from the selected adapter: " + name)
    _, protocol_blob = _regular_bytes(Path(protocol).resolve(), Path(protocol).resolve().parent, max_bytes=1024**2)
    protocol_identity = committed_protocol(protocol, protocol_blob)
    spec = _json(protocol_blob)
    require(spec.get("kind") == "public_same_image_language_contrast" and spec.get("final_test") is False,
            "Unsupported frozen contrast protocol scope")
    comparisons = validate_comparisons(payloads, protocol_blob, dataset)
    production = {name: _regular_bytes(production_root / name, production_root, max_bytes=16 * 1024**2)[1]
                  for name in PROBE_FILES}
    _, probe_blob = _regular_bytes(probe_root / "reload_probe.json", probe_root, max_bytes=1024**2)
    probe = _json(probe_blob)
    image_name = probe["request"]["scene"]["frame"]["image_path"]
    require(image_name in ("probe-image.jpg", "probe-image.png", "probe-image.webp"), "Unsafe probe image path")
    _, image_blob = _regular_bytes(probe_root / image_name, probe_root, max_bytes=32 * 1024**2)
    proof = validate_production_probe(probe, image_blob, production, selected, digest(selected_blob), dataset)
    source_zip, source_manifest = source_archive(run, settings)
    # Only allowlisted public evaluation/model metadata is copied. Never walk the
    # run recursively: optimizer state, private EMT footage and base weights stay out.
    for name in OPTIONAL_RUN_FILES:
        if (run / name).exists(): read(name)
    for name in ADAPTER_FILES:
        if (run / "adapter" / name).exists(): read("adapter/" + name)
    inference = _json(read("inference_config.json"))
    inference_profile(inference, selected, "Saved inference configuration")
    probe_config = _json(production["run.json"])["configuration"]
    for key in ("runtime", "model_id", "revision", "dtype", "quantization", "device", "suffix_layers",
                "inference_resident_layers", "min_pixels", "max_pixels", "max_input_tokens",
                "max_new_tokens", "max_time_s"):
        require(inference.get(key) == probe_config.get(key), "Saved/production inference profile differs: " + key)
    for name, blob in production.items(): payloads[PROBE_FILES[name]] = blob
    payloads["raw_training_result.json"] = payloads["training_result.json"]
    payloads["selected_adapter_manifest.json"] = selected_blob
    payloads["reload_probe.json"], payloads[image_name] = probe_blob, image_blob
    payloads["probe-requests.jsonl"] = (json.dumps(probe["request"], sort_keys=True, allow_nan=False) + "\n").encode()
    payloads["training_source.zip"] = source_zip
    payloads["evaluation/public-contrast-protocol.json"] = protocol_blob
    generated = {"training_source_manifest.json": source_manifest}
    training_config = {"schema_version": 1, "run_kind": "public_destination", "runner": "tools.dropoff_fit_local",
        "base_model_id": STREAMED_MODEL_ID, "base_revision": STREAMED_REVISION,
        "settings": settings, "training_settings_sha256": digest(payloads["training_settings.json"]),
        "scope": "Completed public destination geometry stage; no local clarification labels."}
    generated["training_config.json"] = training_config
    verification = {"verified": True, "fresh_selected_tensor_hashes_match": True,
        "fresh_evaluation_hash_files": ["evaluation/trained.adapter-hashes.json", "evaluation/contrast-trained.adapter-hashes.json"],
        "public_production_probe": proof, "committed_contrast_protocol": protocol_identity,
        "model_quality_verified": False}
    generated["reload_verification.json"] = verification
    result = deepcopy(raw)
    result.update(schema_version=1, run_kind="public_destination", nonzero_updates_verified=True,
        counter_scope="entire_completed_training_schedule",
        changed_adapter_tensors_scope="packaged_selected_checkpoint",
        changed_adapter_tensors=changed, fresh_reload_pending=False, fresh_reload_verified=True,
        free_spatial_comparison_pending=False, local_clarification_training_complete=False,
        raw_training_result_sha256=digest(payloads["raw_training_result.json"]),
        full_schedule={"optimizer_steps": updates, "presentations": presentations, "epochs": epochs},
        packaged_checkpoint={"epoch": epoch, "optimizer_steps": selected["optimizer_steps"],
            "presentations": selected["actual_training_examples_consumed"], "changed_adapter_tensors": changed},
        public_comparisons={name: digest(payloads["evaluation/" + name]) for name in comparisons},
        verification_scope="Real public-destination training, fresh selected-adapter reload and public smoke generation; not final EMT/local clarification acceptance.")
    generated["training_result.json"] = result
    for name, value in generated.items(): payloads[name] = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    staged_manifest = deepcopy(selected)
    flags = {key: True for key in ("nonzero_updates_verified", "fresh_base_reload_verified",
        "generation_smoke_completed", "structured_generation_valid", "spatial_proposal_generated")}
    staged_manifest.update(schema_version=1, run_kind="public_destination", changed_adapter_tensors=changed,
        verification=flags, raw_selected_manifest_sha256=digest(selected_blob),
        training_config_sha256=digest(payloads["training_config.json"]),
        dataset_manifest_sha256=digest(payloads["dataset_manifest.json"]),
        training_result_sha256=digest(payloads["training_result.json"]))
    payloads["adapter/grounding_adapter_manifest.json"] = (json.dumps(staged_manifest, indent=2) + "\n").encode()
    inference.update(model_dir="models/qwen3-vl-8b-local-nf4", cache_dir="models/dropoff-processor-cache/hub", adapter_dir="adapter")
    payloads["inference_config.json"] = (json.dumps(inference, indent=2, allow_nan=False) + "\n").encode()
    completion = {"schema_version": 1, "status": "public_destination_completed",
        "full_schedule": result["full_schedule"], "packaged_checkpoint": result["packaged_checkpoint"],
        "adapter_manifest_sha256": digest(payloads["adapter/grounding_adapter_manifest.json"]),
        "raw_selected_manifest_sha256": digest(selected_blob), "verification": flags,
        "public_development_comparisons_completed": True, "model_quality_verified": False,
        "local_clarification_training_complete": False, "final_EMT_acceptance_complete": False}
    payloads["completion.json"] = (json.dumps(completion, indent=2) + "\n").encode()
    require(all(artifact_allowed(name) for name in payloads),
            "Bundle allowlist needs the new packaging proof/source filenames")
    require(sum(map(len, payloads.values())) <= ARTIFACT_LIMIT, "Staged artifacts exceed the bundle size ceiling")
    out.mkdir(parents=True, exist_ok=False)
    for name, blob in payloads.items():
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream: stream.write(blob)
    return {"out": str(out), "run_kind": "public_destination", "files": len(payloads),
        "full_schedule": result["full_schedule"], "packaged_checkpoint": result["packaged_checkpoint"],
        "model_quality_verified": False, "raw_run_modified": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Completed raw tools.dropoff_fit_local run; never modified")
    parser.add_argument("--production-probe-run", type=Path, required=True,
                        help="tools.dropoff_predict output for the successful public trained-model probe")
    parser.add_argument("--probe-input", type=Path, required=True,
                        help="Directory containing portable reload_probe.json and its public probe-image file")
    parser.add_argument("--protocol", type=Path, required=True,
                        help="Frozen contrast protocol whose exact bytes match both Git HEAD and evaluation")
    parser.add_argument("--out", type=Path, required=True,
                        help="New staging directory outside the raw inputs; bundle it separately")
    args = parser.parse_args(argv)
    try:
        result = stage_run(args.run_dir, args.production_probe_run, args.probe_input, args.protocol, args.out)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"Local handoff staging failed: {error}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
