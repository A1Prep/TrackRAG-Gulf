"""Verify the experimental local adapter and run a matched native interpretation condition."""
import argparse
from collections import Counter
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import zipfile

from tools.dropoff_predict import predict, read_requests
from trackrag.dropoff.data import file_sha256
from trackrag.dropoff.formatting import parse_model_output
from trackrag.dropoff.model import GroundingConfig, GroundingModel, verify_adapter


CONDITIONS = ("local_child", "public_parent", "pretrained")
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors", "grounding_adapter_manifest.json")


def read(path):
    return json.loads(path.read_bytes())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def verify_tensors(parent, child):
    """Read both actual safetensors files into CPU arrays. Reproduce the trainer's dtype/shape/byte hashes."""
    import numpy as np
    from safetensors.numpy import load_file

    config = GroundingConfig.from_mapping(read(child / "inference_config.json"))
    saved = verify_adapter(replace(config, adapter_dir=str((child / "adapter").resolve())))
    parent_config = GroundingConfig.from_mapping(read(parent / "inference_config.json"))
    original = verify_adapter(replace(parent_config, adapter_dir=str((parent / "adapter").resolve())))
    require((saved["base_model_id"], saved["base_revision"]) == (original["base_model_id"], original["base_revision"]),
            "Parent and child do not use the same base model")
    initial = read(child / "initialization-hashes.json")
    parent_binding = read(child / "parent_binding.json")
    manifest = read(child / "adapter/grounding_adapter_manifest.json")
    training = read(child / "training_result.json")
    cohort = read(child / "training_manifest.json")
    require(initial == parent_binding["expected_adapter_parameter_hashes"] and
            manifest["training_manifest_sha256"] == file_sha256(child / "training_manifest.json"),
            "Saved initialization or supervision manifest changed")
    require(manifest["human_reviewed"] is False and training["human_reviewed"] is False and not training["development_used"],
            "This release is specifically the AI-supervised development adapter")
    starts = load_file(str(parent / "adapter/adapter_model.safetensors"))
    ends = load_file(str(child / "adapter/adapter_model.safetensors"))
    require(starts.keys() == ends.keys() and len(ends) == len(initial), "Adapter parameter inventory changed")
    delta_squared, before_squared, n, changed = 0., 0., 0, 0
    for key in sorted(ends):
        a, b = starts[key], ends[key]
        name = key.removesuffix(".weight") + ".default.weight"
        require(key.endswith(".weight") and a.dtype == b.dtype and a.shape == b.shape and
                np.isfinite(a).all() and np.isfinite(b).all(), "Malformed saved adapter tensor")
        def hashed(array):
            return sha256(("torch." + str(array.dtype) + str(tuple(array.shape))).encode() + array.tobytes()).hexdigest()
        require(hashed(a) == initial[name] and hashed(b) == manifest["adapter_parameter_hashes"][name],
                "Saved tensor content differs from the original training record")
        old = a.astype(np.float64)
        delta_squared += float(np.square(b.astype(np.float64) - old).sum())
        before_squared += float(np.square(old).sum())
        n += b.size
        changed += int(not np.array_equal(a, b))
    counts = Counter(training["consumed_case_ids"])
    train_names = {row["request_id"] for row in cohort["rows"]}
    require(set(counts) == train_names and len(counts) == training["train_cases"] and
            all(v == training["epochs_completed"] for v in counts.values()) and
            sum(counts.values()) == training["actual_training_examples_consumed"] and
            changed == training["changed_tensors"] == manifest["changed_adapter_tensors"] and changed > 0,
            "Actual tensor deltas or consumed supervision disagree with the training record")
    return {"kind": "fresh_process_cpu_reload_of_saved_adapter_tensors", "parent_weights_sha256":
            original["files"]["adapter_model.safetensors"]["sha256"], "child_weights_sha256":
            saved["files"]["adapter_model.safetensors"]["sha256"], "adapter_manifest_sha256": saved["manifest_sha256"],
            "base_model_id": config.model_id, "revision": config.revision, "adapter_tensors": len(ends),
            "changed_tensors": changed, "trainable_parameters": n, "delta_l2": delta_squared**.5,
            "relative_l2": (delta_squared / before_squared)**.5 if before_squared else None,
            "actual_presentations": sum(counts.values()), "distinct_train_commands": len(counts),
            "epochs": training["epochs_completed"], "optimizer_steps": training["optimizer_steps"],
            "human_reviewed": False, "training_used_development": False, "full_gpu_generation_tested_here": False}


def check_native_run(run):
    """Verify native receipts at their original paths before reading answers or comparing conditions."""
    meta = read(run / "run.json")
    files = {"input-snapshot.jsonl": "requests_sha256", "resolved-requests.jsonl": "resolved_requests_sha256",
             "receipts.jsonl": "receipts_sha256", "model-outcomes.jsonl": "outcomes_sha256"}
    require(all(file_sha256(run / name) == meta[key] for name, key in files.items()),
            "One or more native prediction-run files no longer match run.json")
    def lines(name):
        return [json.loads(line) for line in (run / name).read_bytes().splitlines() if line.strip()]
    rows, receipts, outputs = (lines(n) for n in ("resolved-requests.jsonl","receipts.jsonl","model-outcomes.jsonl"))
    cfg = GroundingConfig.from_mapping(meta["configuration"])
    expected_adapter = verify_adapter(cfg)
    require(len(rows) == len(receipts) == len(outputs) == meta["requested_rows"] and
            len({r["request_id"] for r in rows}) == len(rows), "Native cohort has missing or duplicate rows")
    checks = []
    for req, receipt, out in zip(rows, receipts, outputs, strict=True):
        source = out["provenance"]
        need_sha = sha256(json.dumps(req, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        require(req["request_id"] == receipt["request_id"] == out["request_id"] and req == receipt["request"]
                and source == receipt["provenance"] and source["request_sha256"] == need_sha and
                source["injected"] is False and source["backend_kind"] == "transformers_local_streamed" and
                source["model_id"] == cfg.model_id and source["revision"] == cfg.revision and
                source["requested_condition"] == ("fine_tuned" if cfg.adapter_dir else "pretrained"),
                "Native request or backend differs from its receipt")
        fr = req["scene"]["frame"]
        image_sha = file_sha256(fr["image_path"])
        require(req["scene"]["source"] == "predicted" and req["scene"]["provenance"]["image_sha256"] == image_sha
                and source["image"]["sha256"] == image_sha, "Native run image is no longer the original input")
        if out["ok"]:
            raw = receipt["raw_text"]
            require(source["adapter"] == expected_adapter and receipt["status"] == "completed" and
                    out["result"] == receipt["result"] == parse_model_output(raw, fr["width"], fr["height"]) and
                    source["backend_details"]["native_fault_isolation"] == "spawn_process" and
                    source["backend_details"]["effective_resident_layers"] == cfg.inference_resident_layers,
                    "Native output, hardware profile or loaded adapter differs from its receipt")
        else:
            require(receipt["status"] == "error" and out["error"] == receipt["error"], "Unacknowledged native failure")
        checks.append({"request_id": req["request_id"], "image_sha256": image_sha, "ok": out["ok"],
                       "request_sha256": need_sha, "raw_text": receipt.get("raw_text"),
                       **({"result": out["result"]} if out["ok"] else {"error": out["error"]})})
    require(sum(not c["ok"] for c in checks) == meta["operational_errors"], "Native error count changed")
    return {"run_sha256": file_sha256(run / "run.json"), "files": {name: meta[key] for name, key in files.items()},
            "requested": len(rows), "operational_errors": meta["operational_errors"], "base_model_id": cfg.model_id,
            "revision": cfg.revision, "adapter_weights_sha256":
                expected_adapter["files"]["adapter_model.safetensors"]["sha256"] if expected_adapter else None,
            "inference_resident_layers": cfg.inference_resident_layers, "rows": checks}


def package(parent, child, output, reload_run=None):
    receipt = verify_tensors(parent, child)
    cfg = read(child / "inference_config.json")
    native = None
    if reload_run is not None:
        native = check_native_run(reload_run)
        run_config = read(reload_run / "run.json")["configuration"]
        require(0 < native["requested"] and native["operational_errors"] < native["requested"] and
                native["adapter_weights_sha256"] == receipt["child_weights_sha256"] and
                {k:v for k,v in run_config.items() if k not in ("adapter_dir","inference_resident_layers")} ==
                {k:v for k,v in cfg.items() if k not in ("adapter_dir","inference_resident_layers")},
                "Reload proof uses another adapter, settings or no successful generation")
        cfg["inference_resident_layers"] = native["inference_resident_layers"]
    cfg["adapter_dir"] = "models/passenger-ai-direct/adapter"
    files = {f"adapter/{name}": (child / "adapter" / name).read_bytes() for name in ADAPTER_FILES}
    files["inference_config.json"] = (json.dumps(cfg, indent=2) + "\n").encode()
    for name in ("training_result.json", "training_manifest.json", "initialization-hashes.json", "training-log.jsonl",
                 "parent_binding.json", "runtime-provenance.json", "training_source.py"):
        files["evidence/" + name] = (child / name).read_bytes()
    files["tensor-verification.json"] = (json.dumps(receipt, indent=2) + "\n").encode()
    if native:
        files["evidence/fresh-native-gpu-reload.json"] = (json.dumps(native, indent=2) + "\n").encode()
    manifest = {"kind": "experimental_native_8b_direct_adapter", "human_reviewed": False,
                "install_under_repo": "models/passenger-ai-direct", "base_distributed_separately": True,
                "files": {name: sha256(data).hexdigest() for name, data in files.items()}}
    files["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in sorted(files.items()):
            z.writestr(zipfile.ZipInfo(name, (2000, 1, 1, 0, 0, 0)), data, compress_type=zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(output) as z:
        require(z.testzip() is None and set(z.namelist()) == files.keys() and all(z.read(k) == v for k, v in files.items()),
                "Packaged bytes changed")
    return dict(receipt, archive_sha256=file_sha256(output), archive_bytes=output.stat().st_size,
                independent_gpu_reload_run_sha256=native["run_sha256"] if native else None)


def run_condition(profiles, condition, requests, output):
    plan = read(profiles / "preflight.json")
    require(set(plan["conditions"]) == set(CONDITIONS) and plan["predictions_generated"] is False
            and plan["summary"]["requests"] == 19 and plan["summary"]["over_budget"] == 0 and
            file_sha256(requests) == plan["requests_sha256"], "Native evaluation must use the frozen matched development bank")
    configs = {name: GroundingConfig.from_mapping(read(profiles / (name + ".json"))) for name in CONDITIONS}
    require(all(replace(c, adapter_dir=None) == configs["pretrained"] for c in configs.values()) and
            configs["pretrained"].adapter_dir is None and all(configs[n].adapter_dir for n in ("local_child","public_parent")),
            "Native comparison settings differ beyond their declared adapter")
    require(set(r["request_id"] for r in read_requests(requests)) == {r["request_id"] for r in plan["rows"]},
            "Evaluation request IDs changed")
    output.parent.mkdir(parents=True, exist_ok=True)
    progress = output.with_name(output.name + "-progress.jsonl")
    if output.exists() or progress.exists():
        raise ValueError("Choose a fresh run destination; previous attempts are preserved")
    with progress.open("x", encoding="utf8", newline="\n") as log:
        class LoggedNativeModel(GroundingModel):
            def interpret_with_receipt(self, request):
                outcome, receipt = super().interpret_with_receipt(request)
                log.write(json.dumps({"request_id": request["request_id"], "ok": outcome["ok"],
                                      "outcome": outcome.get("result") or outcome.get("error")}) + "\n")
                log.flush()
                return outcome, receipt
        result = predict(requests, configs[condition], output, adapter_factory=LoggedNativeModel)
    return {"condition": condition, "source": "native_GroundingModel", "run_sha256": file_sha256(output/"run.json"),
            "requested": result["requested_rows"], "operational_errors": result["operational_errors"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("package", help="Independently verify and package the saved native LoRA, using CPU")
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--child", type=Path, required=True)
    p.add_argument("--zip", type=Path, required=True)
    p.add_argument("--reload-run", type=Path, help="Completed native child run. Package its verified runtime settings and receipt")
    n = commands.add_parser("run", help="Run one of the three matched native conditions; saves the standard evaluator handoff")
    n.add_argument("--profiles", type=Path, required=True)
    n.add_argument("--condition", choices=CONDITIONS, required=True)
    n.add_argument("--requests", type=Path, required=True)
    n.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    result = package(a.parent, a.child, a.zip, a.reload_run) if a.command == "package" else run_condition(a.profiles, a.condition, a.requests, a.out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
