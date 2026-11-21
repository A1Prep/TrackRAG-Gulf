"""Create minimal, hashed GPU source and trained-adapter handoff bundles."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile


MANIFEST = "BUNDLE_MANIFEST.json"
SOURCE_LIMIT = 64 * 1024 * 1024
ARTIFACT_LIMIT = 512 * 1024 * 1024
FORBIDDEN_SOURCE_PARTS = {"__pycache__", ".git", ".venv", ".venv-models", "venv", "env", "data", "models", "runs"}
# These run in the full EMT checkout before uploading prepared requests. Their
# Task 1/2 dependencies are deliberately absent from the minimal training ZIP.
FULL_CHECKOUT_ONLY = {
    "tools/dropoff_prepare_scene.py", "tests/test_dropoff_prepare_scene.py",
}
REQUIRED_SOURCE = {
    "trackrag/contracts.py", "trackrag/dropoff/__init__.py",
    "trackrag/dropoff/contracts.py", "trackrag/dropoff/data.py",
    "trackrag/dropoff/formatting.py", "trackrag/dropoff/model.py",
    "tools/dropoff_train.py", "tools/dropoff_bootstrap_regseg.py",
    "tools/dropoff_bundle.py", "configs/dropoff/training-pilot.json",
    "configs/dropoff/regseg-scene-guards.json", "requirements-grounding-train.txt",
}
SOURCE_PROTOCOLS = {
    "experiments/dropoff/public-contrast-protocol-20261121.json",
    "experiments/dropoff/public-prompt-ablation-protocol-20261121.json",
    "experiments/dropoff/public-target-encoding-protocol-20261121.json",
}
TEST_SUPPORT_NOTEBOOK = "notebooks/dropoff_gpu_pilot.ipynb"
RUN_FILES = {
    "training_config.json", "dataset_manifest.json", "training_result.json",
    "preflight.json", "reload_probe.json", "reload_verification.json",
    "completion.json", "failure.json", "inference_config.json",
    "environment_freeze.txt", "cloud_handoff.json", "trainer_state.json",
    "cloud_reload_stdout.txt", "cloud_reload_stderr.txt",
    "probe-image.jpg", "probe-image.png", "probe-image.webp",
    "probe-requests.jsonl",
    "training_settings.json", "evaluation_cohort.json", "runtime_provenance.json",
    "target-encoding.json",
    "hardware.json", "training-log.jsonl", "raw_training_result.json",
    "selected_adapter_manifest.json", "pretrained_teacher_metrics.json",
    "epoch-1-teacher-metrics.json", "epoch-2-teacher-metrics.json",
    "evaluation/pretrained.jsonl", "evaluation/pretrained.summary.json",
    "evaluation/pretrained.teacher-metrics.json", "evaluation/trained.jsonl",
    "evaluation/pretrained.target-encoding.json", "evaluation/trained.target-encoding.json",
    "evaluation/contrast-pretrained.target-encoding.json", "evaluation/contrast-trained.target-encoding.json",
    "evaluation/trained.summary.json", "evaluation/trained.teacher-metrics.json",
    "evaluation/trained.adapter-hashes.json", "evaluation/comparison.json",
    "evaluation/contrast-pretrained.jsonl", "evaluation/contrast-pretrained.summary.json",
    "evaluation/contrast-trained.jsonl", "evaluation/contrast-trained.summary.json",
    "evaluation/contrast-trained.adapter-hashes.json", "evaluation/contrast-comparison.json",
    "evaluation/public-contrast-protocol.json", "software-validation.json",
    "production_probe_summary.json", "production_probe_outcomes.jsonl",
    "production_probe_receipts.jsonl", "production_probe_requests.jsonl",
    "production_probe_input_snapshot.jsonl", "training_source_manifest.json",
    "training_source.zip",
}
ADAPTER_FILES = {
    "adapter_model.safetensors", "adapter_config.json", "grounding_adapter_manifest.json",
    "preprocessor_config.json", "processor_config.json", "tokenizer_config.json",
    "special_tokens_map.json", "tokenizer.json", "added_tokens.json",
    "vocab.json", "merges.txt", "chat_template.jinja", "chat_template.json", "README.md",
}
REQUIRED_ARTIFACT = {
    "adapter/adapter_model.safetensors", "adapter/adapter_config.json",
    "adapter/grounding_adapter_manifest.json", "training_config.json",
    "dataset_manifest.json", "training_result.json", "reload_probe.json",
    "reload_verification.json", "completion.json", "inference_config.json",
}


class BundleError(ValueError):
    pass


def _hash(blob):
    return hashlib.sha256(blob).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BundleError(f"Duplicate manifest field: {key}")
        result[key] = value
    return result


def _json(blob):
    def invalid(value):
        raise BundleError(f"Nonfinite JSON constant: {value}")
    try:
        return json.loads(blob.decode("utf-8"), object_pairs_hook=_unique, parse_constant=invalid)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BundleError(f"Invalid UTF-8 JSON: {error}") from error


def _validate_unexecuted_notebook(blob, *, load_json=_json):
    """Validate the optional Python test-support notebook without executing it."""
    try:
        notebook = load_json(blob)
        if (not isinstance(notebook, dict) or type(notebook.get("nbformat")) is not int
                or notebook["nbformat"] != 4 or type(notebook.get("nbformat_minor")) is not int
                or notebook["nbformat_minor"] < 0 or not isinstance(notebook.get("metadata"), dict)
                or not isinstance(notebook.get("cells"), list)):
            raise ValueError("Unsupported notebook structure")
        identifiers = set()
        for cell in notebook["cells"]:
            if (not isinstance(cell, dict) or cell.get("cell_type") not in ("code", "markdown", "raw")
                    or not isinstance(cell.get("metadata"), dict)):
                raise ValueError("Invalid notebook cell")
            source = cell.get("source")
            if not (isinstance(source, str) or isinstance(source, list) and all(isinstance(s, str) for s in source)):
                raise ValueError("Notebook source must be text")
            if notebook["nbformat_minor"] >= 5:
                identity = cell.get("id")
                if (not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", identity)
                        or identity in identifiers):
                    raise ValueError("Invalid or duplicate notebook cell ID")
                identifiers.add(identity)
            if cell["cell_type"] == "code":
                if ("execution_count" not in cell or cell["execution_count"] is not None
                        or cell.get("outputs") != []):
                    raise ValueError("Notebook must be unexecuted, with empty outputs and null execution_count")
                ast.parse(source if isinstance(source, str) else "".join(source))
            elif "outputs" in cell or "execution_count" in cell:
                raise ValueError("Non-code notebook cell contains execution data")
    except (ValueError, TypeError, KeyError, SyntaxError) as error:
        raise BundleError("Invalid or executed test-support notebook: " + str(error)) from error


def safe_zip_path(name):
    if not isinstance(name, str):
        raise BundleError("ZIP paths must be text")
    path = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or "\x00" in name
            or path.is_absolute() or ".." in path.parts or name != path.as_posix()):
        raise BundleError(f"Unsafe/noncanonical ZIP path: {name!r}")
    return path


def source_allowed(name):
    path = safe_zip_path(name)
    parts = path.parts
    if name in FULL_CHECKOUT_ONLY:
        return False
    if any(part.startswith(".") or part in FORBIDDEN_SOURCE_PARTS for part in parts):
        return False
    if name in SOURCE_PROTOCOLS:
        return True
    if name == TEST_SUPPORT_NOTEBOOK:
        return True
    if name in ("trackrag/__init__.py", "trackrag/contracts.py", "requirements-grounding-train.txt", "requirements-grounding-local.txt", "requirements-grounding-format.txt"):
        return True
    if len(parts) >= 3 and parts[:2] == ("trackrag", "dropoff"):
        return path.suffix == ".py" and "__pycache__" not in parts
    if len(parts) == 2 and parts[0] == "tools":
        return parts[1].startswith("dropoff_") and path.suffix == ".py"
    if len(parts) == 3 and parts[:2] == ("configs", "dropoff"):
        return path.suffix == ".json"
    if len(parts) == 2 and parts[0] == "docs":
        return parts[1].startswith("dropoff") and path.suffix == ".md"
    if len(parts) == 2 and parts[0] == "tests":
        return parts[1].startswith("test_dropoff") and path.suffix == ".py"
    return False


def artifact_allowed(name):
    path = safe_zip_path(name)
    return name in RUN_FILES or (
        len(path.parts) == 2 and path.parts[0] == "adapter" and path.parts[1] in ADAPTER_FILES
    )


def _regular_bytes(path, root, *, max_bytes=None):
    """Reject symlinks/junctions in the source chain, even if they resolve inside."""
    path = Path(path).absolute()
    root = Path(root).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise BundleError("Source path is outside the selected root") from error
    cursor = root
    for part in relative.parts:
        cursor /= part
        details = cursor.lstat()
        if stat.S_ISLNK(details.st_mode) or getattr(details, "st_file_attributes", 0) & 0x400:
            raise BundleError(f"Symlink/reparse source is forbidden: {relative}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or not stat.S_ISREG(path.lstat().st_mode):
        raise BundleError(f"Source is not a regular contained file: {relative}")
    if max_bytes is not None and path.stat().st_size > max_bytes:
        raise BundleError(f"File exceeds remaining bundle byte ceiling: {relative}")
    return relative.as_posix(), path.read_bytes()


def _source_files(root, include_docs, include_tests):
    root = Path(root).resolve()
    candidates = [root / name for name in REQUIRED_SOURCE]
    # The current repository uses a top-level namespace package. Include an
    # initializer when a future checkout has one, but do not invent one.
    initializer = root / "trackrag/__init__.py"
    if initializer.exists() or initializer.is_symlink():
        candidates.append(initializer)
    for requirements_name in ('requirements-grounding-local.txt', 'requirements-grounding-format.txt'):
        local_requirements = root / requirements_name
        if local_requirements.exists() or local_requirements.is_symlink():
            candidates.append(local_requirements)
    candidates.extend(root/name for name in SOURCE_PROTOCOLS if (root/name).exists())
    dropoff = root / "trackrag/dropoff"
    if dropoff.is_dir():
        for parent, dirs, files in os.walk(dropoff, followlinks=False):
            for directory in dirs:
                if directory.startswith(".") or directory in FORBIDDEN_SOURCE_PARTS:
                    continue
                candidate = Path(parent, directory)
                if candidate.is_symlink() or getattr(candidate.lstat(), "st_file_attributes", 0) & 0x400:
                    raise BundleError("Symlink/reparse directory under allowlisted source")
            dirs[:] = [name for name in dirs if not name.startswith(".") and name not in FORBIDDEN_SOURCE_PARTS]
            candidates.extend(Path(parent, name) for name in files if name.endswith(".py") and not name.startswith("."))
    for pattern in ("tools/dropoff_*.py", "configs/dropoff/*.json"):
        candidates.extend(path for path in root.glob(pattern) if source_allowed(path.relative_to(root).as_posix()))
    if include_docs:
        candidates.extend(root.glob("docs/dropoff*.md"))
    if include_tests:
        candidates.extend(path for path in root.glob("tests/test_dropoff*.py")
                          if source_allowed(path.relative_to(root).as_posix()))
        notebook = root / TEST_SUPPORT_NOTEBOOK
        if notebook.exists() or notebook.is_symlink():
            candidates.append(notebook)
    contents = {}
    for path in sorted(set(candidates)):
        if not path.exists() and not path.is_symlink():
            raise BundleError(f"Required source file is missing: {path.relative_to(root)}")
        name, blob = _regular_bytes(path, root, max_bytes=SOURCE_LIMIT - sum(map(len, contents.values())))
        if not source_allowed(name):
            raise BundleError(f"Source is outside the explicit allowlist: {name}")
        if name == TEST_SUPPORT_NOTEBOOK:
            _validate_unexecuted_notebook(blob)
        contents[name] = blob
    return contents


def _write_bundle(contents, output, kind, metadata):
    if not contents:
        raise BundleError("Cannot package an empty bundle")
    limit = SOURCE_LIMIT if kind == "source" else ARTIFACT_LIMIT
    if sum(map(len, contents.values())) > limit:
        raise BundleError("Bundle exceeds its uncompressed byte ceiling")
    names = list(contents)
    if len(names) > 1023:
        raise BundleError("Too many bundle files")
    if len({name.casefold() for name in names}) != len(names):
        raise BundleError("Case-colliding bundle paths")
    manifest = {
        "schema_version": 1, "bundle_kind": kind,
        "files": {name: {"bytes": len(blob), "sha256": _hash(blob)}
                  for name, blob in sorted(contents.items())},
        "metadata": metadata,
    }
    manifest_blob = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    output = Path(output).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with output.open("xb") as raw:
            created = True
            with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, blob in sorted({**contents, MANIFEST: manifest_blob}.items()):
                    safe_zip_path(name)
                    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = (stat.S_IFREG | 0o644) << 16
                    archive.writestr(info, blob)
    except Exception:
        if created:
            output.unlink(missing_ok=True)  # Only the just-created exclusive output file.
        raise
    verified = verify_bundle(output)
    return {"output": str(output), "sha256": _hash(output.read_bytes()),
            "files": len(contents), "bundle_kind": kind, "verified": verified["verified"]}


def bundle_source(root, output, *, include_docs=True, include_tests=False):
    return _write_bundle(
        _source_files(root, include_docs, include_tests), output, "source",
        {"purpose": "minimal_grounding_source_for_gpu_execution",
         "model_weights_included": False, "credentials_included": False,
         "actual_8b_gpu_execution_verified": False},
    )


def _artifact_completion(contents):
    missing = REQUIRED_ARTIFACT - contents.keys()
    if missing:
        return False, f"Missing required artifacts: {sorted(missing)}"
    adapter = _json(contents["adapter/grounding_adapter_manifest.json"])
    config = _json(contents["training_config.json"])
    peft = _json(contents["adapter/adapter_config.json"])
    training = _json(contents["training_result.json"])
    if any(not isinstance(value, dict) for value in (adapter, config, peft, training)):
        return False, "Model/training provenance must contain JSON objects"
    if (type(adapter.get("schema_version")) is not int or adapter["schema_version"] != 1
            or adapter.get("artifact_kind") != "lora_adapter"
            or type(config.get("schema_version")) is not int or config["schema_version"] != 1
            or type(training.get("schema_version")) is not int or training["schema_version"] != 1):
        return False, "Missing or unsupported adapter/training schema or artifact kind"
    base, revision = adapter.get("base_model_id"), adapter.get("base_revision")
    if (not isinstance(base, str) or not base.strip()
            or not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision)
            or config.get("base_model_id") != base or config.get("base_revision") != revision
            or peft.get("base_model_name_or_path") != base or peft.get("revision") != revision
            or peft.get("peft_type") != "LORA"):
        return False, "Base-model identity/revision differs across adapter, PEFT and training configuration"
    run_kind = config.get("run_kind")
    if run_kind not in ("pilot", "adaptation", "public_destination") or adapter.get("run_kind") != run_kind or training.get("run_kind") != run_kind:
        return False, "Run kind is missing or inconsistent"
    if (training.get("nonzero_updates_verified") is not True
            or type(training.get("optimizer_steps")) is not int or training["optimizer_steps"] < 1
            or type(training.get("changed_adapter_tensors")) is not int or training["changed_adapter_tensors"] < 1):
        return False, "Training result lacks verified nonzero updates and positive optimizer steps"
    expected = adapter.get("files")
    if not isinstance(expected, dict):
        return False, "Adapter manifest lacks file hashes"
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if expected.get(name) != _hash(contents["adapter/" + name]):
            return False, f"Adapter manifest hash mismatch: {name}"
    for key, name in (
        ("training_config_sha256", "training_config.json"),
        ("dataset_manifest_sha256", "dataset_manifest.json"),
        ("training_result_sha256", "training_result.json"),
    ):
        if adapter.get(key) != _hash(contents[name]):
            return False, f"Adapter provenance hash mismatch: {name}"
    completion = _json(contents["completion.json"])
    reload = _json(contents["reload_verification.json"])
    if not isinstance(completion, dict) or not isinstance(reload, dict):
        return False, "Completion/reload records must be JSON objects"
    verification = adapter.get("verification", {})
    if not isinstance(verification, dict):
        return False, "Adapter verification flags must be an object"
    required_flags = (
        "nonzero_updates_verified", "fresh_base_reload_verified",
        "generation_smoke_completed", "structured_generation_valid", "spatial_proposal_generated",
    )
    if (completion.get("status") != run_kind + "_completed"
            or reload.get("verified") is not True
            or any(verification.get(key) is not True for key in required_flags)):
        return False, "Run has not passed genuine-update/fresh-reload/spatial-generation checks"
    if "cloud_handoff.json" in contents:
        cloud = _json(contents["cloud_handoff.json"])
        fresh = cloud.get("fresh_process_reload") if isinstance(cloud, dict) else None
        if (not isinstance(cloud, dict) or type(cloud.get("training_exit_code")) is not int
                or cloud["training_exit_code"] != 0 or not isinstance(fresh, dict)
                or fresh.get("status") != "passed" or type(fresh.get("exit_code")) is not int
                or fresh["exit_code"] != 0 or fresh.get("verified") is not True):
            return False, "Cloud training/fresh-process reload is incomplete or failed; stale inner receipts cannot complete it"
    probe = _json(contents["reload_probe.json"])
    if not isinstance(probe, dict) or not isinstance(probe.get("request"), dict):
        return False, "Portable reload probe request is malformed"
    scene = probe["request"].get("scene")
    frame = scene.get("frame") if isinstance(scene, dict) else None
    image_name = frame.get("image_path") if isinstance(frame, dict) else None
    if image_name not in ("probe-image.jpg", "probe-image.png", "probe-image.webp"):
        return False, "Reload probe is not bound to a portable public probe image"
    if image_name not in contents or probe.get("image_sha256") != _hash(contents[image_name]):
        return False, "Portable reload probe image is missing or changed"
    if sum(name in contents for name in ("probe-image.jpg", "probe-image.png", "probe-image.webp")) != 1:
        return False, "A completed handoff must contain exactly one matching portable probe image"
    return True, "Artifact completeness and hashes verified; this is not model-quality validation"


def bundle_artifacts(run_dir, output, *, allow_incomplete=False):
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise BundleError("Training run directory is unavailable")
    contents = {}
    for name in sorted(RUN_FILES | {"adapter/" + item for item in ADAPTER_FILES}):
        path = root / name
        if path.exists() or path.is_symlink():
            actual_name, blob = _regular_bytes(path, root, max_bytes=ARTIFACT_LIMIT - sum(map(len, contents.values())))
            if not artifact_allowed(actual_name):
                raise BundleError("Artifact is outside the explicit allowlist")
            contents[actual_name] = blob
    try:
        complete, reason = _artifact_completion(contents)
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        complete, reason = False, f"Invalid run verification record: {type(error).__name__}: {error}"
    if not complete and not allow_incomplete:
        raise BundleError(reason + "; use --allow-incomplete only to export labelled diagnostics")
    return _write_bundle(contents, output, "adapter" if complete else "diagnostic", {
        "run_artifacts_complete": complete, "status_detail": reason,
        "model_quality_verified": False, "base_weights_included": False,
        "bulk_dataset_included": False, "portable_public_reload_probe_included": complete,
    })


def verify_bundle(path):
    """Verify every manifest entry, ZIP path/type, size and SHA before use."""
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(names) > 1024 or len(set(names)) != len(names):
            raise BundleError("Duplicate or excessive ZIP entries")
        if len({name.casefold() for name in names}) != len(names):
            raise BundleError("Case-colliding ZIP entries")
        for entry in entries:
            safe_zip_path(entry.filename)
            if entry.orig_filename != entry.filename:
                raise BundleError("Truncated/noncanonical ZIP member name")
            mode = entry.external_attr >> 16
            if entry.is_dir() or stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG)):
                raise BundleError("Non-regular ZIP member")
            if entry.flag_bits & 1:
                raise BundleError("Encrypted ZIP is unsupported")
        if MANIFEST not in names:
            raise BundleError("Bundle manifest is missing")
        if archive.getinfo(MANIFEST).file_size > 1024 * 1024:
            raise BundleError("Bundle manifest is oversized")
        manifest = _json(archive.read(MANIFEST))
        if (not isinstance(manifest, dict) or set(manifest) != {"schema_version", "bundle_kind", "files", "metadata"}
                or manifest["schema_version"] != 1 or type(manifest["schema_version"]) is not int):
            raise BundleError("Unsupported bundle manifest schema")
        kind = manifest["bundle_kind"]
        if kind not in ("source", "adapter", "diagnostic"):
            raise BundleError("Unknown bundle kind")
        files = manifest["files"]
        if not isinstance(files, dict) or not files or set(names) != set(files) | {MANIFEST}:
            raise BundleError("ZIP contents differ from the manifest")
        if kind == "source" and not REQUIRED_SOURCE <= files.keys():
            raise BundleError("Source bundle is missing required training files")
        if kind == "adapter" and not REQUIRED_ARTIFACT <= files.keys():
            raise BundleError("Adapter bundle is missing required run artifacts")
        limit = SOURCE_LIMIT if kind == "source" else ARTIFACT_LIMIT
        if sum(entry.file_size for entry in entries) > limit + 1024 * 1024:
            raise BundleError("Bundle exceeds uncompressed byte ceiling")
        allowed = source_allowed if kind == "source" else artifact_allowed
        contents = {}
        for name, expected in files.items():
            if not allowed(name) or not isinstance(expected, dict) or set(expected) != {"bytes", "sha256"}:
                raise BundleError("Unapproved bundle file or manifest fields")
            if type(expected["bytes"]) is not int or expected["bytes"] < 0:
                raise BundleError("Invalid declared file size")
            if not isinstance(expected["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", expected["sha256"]):
                raise BundleError("Invalid declared SHA256")
            if archive.getinfo(name).file_size != expected["bytes"]:
                raise BundleError(f"File size mismatch: {name}")
            blob = archive.read(name)
            if len(blob) != expected["bytes"] or _hash(blob) != expected["sha256"]:
                raise BundleError(f"File SHA256 mismatch: {name}")
            if kind == "source" and name == TEST_SUPPORT_NOTEBOOK:
                _validate_unexecuted_notebook(blob)
            if kind == "adapter":
                contents[name] = blob
        if kind == "adapter":
            complete, reason = _artifact_completion(contents)
            if not complete:
                raise BundleError(reason)
    return {"verified": True, "bundle_kind": kind, "files": len(files), "metadata": manifest["metadata"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    source = commands.add_parser("source")
    source.add_argument("--repo-root", type=Path, default=Path("."))
    source.add_argument("--out", type=Path, required=True)
    source.add_argument("--include-tests", action="store_true")
    source.add_argument("--no-docs", action="store_true")
    artifacts = commands.add_parser("artifacts")
    artifacts.add_argument("--run-dir", type=Path, required=True)
    artifacts.add_argument("--out", type=Path, required=True)
    artifacts.add_argument("--allow-incomplete", action="store_true")
    verify = commands.add_parser("verify")
    verify.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "source":
            result = bundle_source(args.repo_root, args.out, include_docs=not args.no_docs, include_tests=args.include_tests)
        elif args.command == "artifacts":
            result = bundle_artifacts(args.run_dir, args.out, allow_incomplete=args.allow_incomplete)
        else:
            result = verify_bundle(args.bundle)
    except (BundleError, OSError, zipfile.BadZipFile) as error:
        parser.exit(2, f"Grounding bundle failed: {error}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
