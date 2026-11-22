"""Package the separately measured native clarification adapter and its original local receipts."""

import argparse
from collections import Counter
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from tools.dropoff_ai_acceptance import ADAPTER_FILES
from tools.dropoff_import_task2_review import RECORDS, REVISION
from tools.dropoff_reviewed_trial import score
from trackrag.dropoff.data import file_sha256
from trackrag.dropoff.model import GroundingConfig, verify_adapter


INSTALL = "models/passenger-reviewed-clarification"
PROTOCOL_SHA256 = "702c5f5575431df621b0e78847ddadb67d102e487a4be44016050773e4af8b81"
NATIVE_FILES = {
    "run.json", "input-snapshot.jsonl", "resolved-requests.jsonl", "model-outcomes.jsonl", "receipts.jsonl",
}
TRAINING_FILES = {
    "training_result.json", "training_settings.json", "training-log.jsonl", "dataset_manifest.json",
    "initialization-hashes.json", "parent-binding.json", "target-encoding.json", "runtime-provenance.json",
    "parent-development-nll.json", "epoch-1-development-nll.json", "epoch-2-development-nll.json",
}


def read(path):
    return json.loads(path.read_bytes())


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def check_training(child):
    training = read(child / "training_result.json")
    dataset = read(child / "dataset_manifest.json")
    for split in ("train", "development"):
        expected_ids = RECORDS[split]
        expected_groups = {"emt-recording:" + case_id.split("-")[-2] for case_id in expected_ids}
        if (Counter(dataset[split + "_case_ids"]) != Counter(expected_ids)
                or Counter(dataset[split + "_scene_groups"]) != Counter(expected_groups)):
            raise ValueError("Training manifest has a different nonfinal case or scene grouping")
    if (training["coverage"] != "clarification-only"
            or training["actual_training_completed"] is not True or training["injected_runtime"] is not False
            or training["training_examples_unique"] != 4 or training["examples_consumed"] != 8
            or training["epochs_completed"] != 2 or training["optimizer_steps"] != 4
            or Counter(training["actual_training_case_ids"]) != {k: 2 for k in RECORDS["train"]}):
        raise ValueError("Expected the recorded 4 × 2 nonfinal human-confirmed training schedule")
    logs = [json.loads(line) for line in (child / "training-log.jsonl").read_bytes().splitlines()]
    for epoch in (1, 2):
        rows = [r for r in logs if r["epoch"] == epoch]
        if Counter(r["case_id"] for r in rows) != Counter(RECORDS["train"]):
            raise ValueError("A training epoch differs from the published four-case cohort")
    if len(logs) != 8:
        raise ValueError("Unexpected additional training log entries")
    return training


def package(parent, study, result_path, out):
    child, trial = study / "child", study / "trial"
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if file_sha256(trial / "protocol.json") != PROTOCOL_SHA256:
        raise ValueError("Native trial differs from its predeclared protocol")
    training = check_training(child)
    recorded = read(result_path)
    with TemporaryDirectory(dir=out.parent) as temporary:
        temp = Path(temporary)
        # Recompute the four-field scores and the parent/child tensor comparison from the native artifacts.
        score(parent, child, trial, temp / "rescore.json")
        if read(temp / "rescore.json") != recorded or recorded["source_revision"] != REVISION:
            raise ValueError("Published trial scores do not reproduce from the observed GPU runs")
        weights = recorded["tensor_verification"]
        for condition, details in recorded["fresh_generation"].items():
            if details["operational_errors"] >= len(details["rows"]):
                raise ValueError(f"No successful fresh native generation for {condition}")
        actual_cfg = read(trial / "human_child/run.json")["configuration"]
        deployed = dict(actual_cfg, adapter_dir=INSTALL + "/adapter")
        for key in ("model_dir", "cache_dir"):
            directory = Path(deployed[key])
            if directory.is_absolute() or directory.drive or ".." in directory.parts:
                raise ValueError(f"Expected a repository-relative {key}")
        files = {
            "inference_config.json": json_bytes(deployed),
            "evidence/trial-result.json": result_path.read_bytes(),
            "evidence/protocol.json": (trial / "protocol.json").read_bytes(),
            "evidence/source.json": (study / "data/source.json").read_bytes(),
        }
        for name in ADAPTER_FILES:
            files["adapter/" + name] = (child / "adapter" / name).read_bytes()
        for name in TRAINING_FILES:
            files["evidence/training/" + name] = (child / name).read_bytes()
        for condition in ("human_child", "public_parent"):
            for name in NATIVE_FILES:
                files[f"evidence/native/{condition}/{name}"] = (trial / condition / name).read_bytes()
        actual_sha = sha256(files["adapter/adapter_model.safetensors"]).hexdigest()
        if actual_sha != weights["child_sha256"]:
            raise ValueError("Packaged adapter differs from the freshly scored child")
        public_evidence = Path(__file__).resolve().parents[1] / "experiments/dropoff"
        for name in ("passenger-human-six-independent-audit-20261122.json",
                     "passenger-human-six-source-pixels-20261122.json"):
            files["evidence/" + name] = (public_evidence / name).read_bytes()
        audit = json.loads(files["evidence/passenger-human-six-source-pixels-20261122.json"])
        independent = json.loads(files["evidence/passenger-human-six-independent-audit-20261122.json"])
        source = json.loads(files["evidence/source.json"])
        verified = {r["jpeg_sha256"] for r in audit["verified_nonfinal_source_images"]}
        if (source["git_revision"] != REVISION or len(verified) != 6
                or set(source["images"].values()) != verified
                or audit["peer_package_audit_sha256"] != sha256(
                    files["evidence/passenger-human-six-independent-audit-20261122.json"]).hexdigest()):
            raise ValueError("Peer reviews and independent source-pixel verification disagree")
        datasets = json.loads(files["evidence/training/dataset_manifest.json"])
        bindings = json.loads(files["evidence/training/target-encoding.json"])["cases"]
        originals = independent["nonfinal"]["original_approved_cases"]
        artifacts = {row["case_id"]: row for row in datasets["case_artifacts"]}
        if not set(artifacts) == set(bindings) == set(originals):
            raise ValueError("The training source and two independent confirmation records differ in membership")
        for identity, original in originals.items():
            if (artifacts[identity]["image_sha256"] != original["jpeg_sha256"]
                    or bindings[identity]["input_sha256"] != original["source_review_input_sha256"]
                    or bindings[identity]["original_supervision_sha256"] != original["reviewed_four_field_sha256"]):
                raise ValueError("A training image, inference input or four-field target has a different binding")
        metadata = {
            "kind": "native_direct_8b_human_confirmed_clarification_experiment",
            "install_under_repository_root": INSTALL,
            "study_scope": "Four training frames and two development frames share one AI-assisted instruction "
                           "and clarification, with two confirmations recorded by Task 2. The two additional "
                           "paraphrases and two controls have separate reference types in the native score.",
            "base_and_processor_distributed_separately": True,
            "original_images_included": False,
            "raw_native_references": "The original request paths identify the machine that measured these "
                                     "runs. Running fresh requests requires access to their original images.",
            "source_revision": REVISION,
            "protocol_sha256": PROTOCOL_SHA256,
            "weights_sha256": actual_sha,
            "files": {n: sha256(b).hexdigest() for n, b in sorted(files.items())},
        }
        files["manifest.json"] = json_bytes(metadata)
        staged = temp / "verified.zip"
        with ZipFile(staged, "x") as archive:
            for name, data in sorted(files.items()):
                member = ZipInfo(name, (2000, 1, 1, 0, 0, 0))
                archive.writestr(member, data, compress_type=ZIP_DEFLATED)
        with ZipFile(staged) as archive:
            if (archive.testzip() or set(archive.namelist()) != set(files)
                    or any(archive.read(k) != v for k, v in files.items())):
                raise ValueError("Native evidence archive failed its byte-for-byte readback")
            installation = temp / INSTALL
            (installation / "adapter").mkdir(parents=True)
            for name in ADAPTER_FILES:
                (installation / "adapter" / name).write_bytes(archive.read("adapter/" + name))
            loaded = GroundingConfig.from_mapping(json.loads(archive.read("inference_config.json")))
            verified = verify_adapter(replace(loaded, adapter_dir=str(installation / "adapter")))
            if verified["files"]["adapter_model.safetensors"]["sha256"] != actual_sha:
                raise ValueError("Installed native weights differ from the real evaluated adapter")
        # Publish only a fully verified archive and never replace an existing release.
        os.link(staged, out)
    return {
        "kind": metadata["kind"], "asset": out.name, "sha256": file_sha256(out), "bytes": out.stat().st_size,
        "entries_reread_from_archive": len(files), "install_under_repository_root": INSTALL,
        "native_score_sha256": file_sha256(result_path), "protocol_sha256": PROTOCOL_SHA256,
        "tensors": weights, "human_training_presentations": training["examples_consumed"],
        "optimizer_steps": training["optimizer_steps"], "selected_checkpoint": training["selected_checkpoint"],
        "native_condition_run_sha256": {k: v["run_sha256"] for k, v in recorded["fresh_generation"].items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "study", "result", "out", "receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    result = package(args.parent, args.study, args.result, args.out)
    with args.receipt.open("xb") as stream:
        stream.write(json_bytes(result))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
