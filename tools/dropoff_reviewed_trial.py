"""Prepare or score the matched local clarification trial on its exact reviewed inputs."""

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re

from tools.dropoff_ai_acceptance import check_native_run
from tools.dropoff_import_task2_review import RECORDS, REVISION
from tools.dropoff_predict import read_requests
from trackrag.dropoff.data import file_sha256, load_cases


PARAPHRASES = ("Could you let me out alongside the vehicle straight ahead?",
               "I'd like to get out next to the car in front of us, please.")
CONTROLS = {"passenger-b07": "proposed", "passenger-b10": "insufficient_visual_evidence"}


def save(path, obj):
    with path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(obj, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")


def prepare(data, control_requests, out):
    source = json.loads((data / "source.json").read_bytes())
    if source["git_revision"] != REVISION or source["reserved_rows_materialized"] != 0:
        raise ValueError("Incorrect review source")
    dev = sorted(load_cases(data / "development.jsonl", allowed_partitions={"development"}),
                 key=lambda r: r["case_id"])
    if {r["case_id"] for r in dev} != RECORDS["development"] or len(dev) != len(PARAPHRASES):
        raise ValueError("Unexpected development cohort")
    rows, targets = [], {}
    for case, paraphrase in zip(dev, PARAPHRASES, strict=True):
        req = deepcopy(case["request"])
        req["scene"].setdefault("provenance", {})["image_sha256"] = case["source"]["image_sha256"]
        expectation = case["target"]
        if expectation["status"] != "needs_clarification":
            raise ValueError("Unexpected reviewed decision")
        rows.append(req)
        targets[req["request_id"]] = {"group": "two_reviewer_confirmations_same_wording", "target": expectation}
        variation = deepcopy(req)
        variation.update(request_id=req["request_id"] + "-paraphrase", text=paraphrase)
        rows.append(variation)
        targets[variation["request_id"]] = {"group": "unreviewed_paraphrase_same_scene",
                                            "target": {"status": "needs_clarification"}}
    controls = [r for r in read_requests(control_requests) if r["request_id"] in CONTROLS]
    if {r["request_id"] for r in controls} != set(CONTROLS) or len(controls) != len(CONTROLS):
        raise ValueError("Expected both earlier labelled control requests")
    for req in controls:
        rows.append(req)
        targets[req["request_id"]] = {"group": "existing_ai_development_control",
                                      "target": {"status": CONTROLS[req["request_id"]]}}
    out.mkdir(parents=True, exist_ok=False)
    requests = out / "requests.jsonl"
    requests.write_bytes(b"".join((json.dumps(r, sort_keys=True, allow_nan=False) + "\n").encode() for r in rows))
    save(out / "protocol.json", {"source_revision": REVISION, "source_sha256": file_sha256(data / "source.json"),
        "requests_sha256": file_sha256(requests), "control_source_sha256": file_sha256(control_requests),
        "hypothesis": "A clarification-only trial can improve native decisions on the reviewed repeated phrase. "
                      "The two same-scene paraphrases and two existing controls probe specificity.",
        "training": {"epochs": 2, "accumulation": 2, "learning_rate": 5e-5, "seed": 42},
        "selection": "Lowest development assistant-token negative log likelihood among the two epochs. "
                     "Development measures fit and checkpoint selection on two photographs with repeated language.",
        "inference": {"max_time_s": 300, "max_input_tokens": 3072, "inference_resident_layers": 2,
                      "observation_format": "compact_rows_v1"},
        "targets_scoring_only": targets,
        "default_release": "Retain the independently evaluated passenger adapter. This single-phrase cohort "
                           "cannot establish a replacement for the full grounding task."})


def tensor_comparison(parent, child):
    from safetensors.numpy import load_file
    import numpy as np

    first, second = [load_file(str(d / "adapter/adapter_model.safetensors")) for d in (parent, child)]
    initial = json.loads((child / "initialization-hashes.json").read_bytes())
    if not initial["matches_verified_parent"] or initial["injected_runtime"]:
        raise ValueError("Training did not initialize from the verified public checkpoint")
    expected = json.loads((child / "adapter/grounding_adapter_manifest.json").read_bytes())["adapter_parameter_hashes"]
    if set(first) != set(second) or len(expected) != len(second):
        raise ValueError("LoRA tensor inventory changed")
    changed = parameters = 0
    for key, before in first.items():
        after = second[key]
        if before.shape != after.shape or before.dtype != after.dtype or not np.isfinite(after).all():
            raise ValueError("Malformed saved LoRA parameters")
        name = key.removesuffix(".weight") + ".default.weight"
        for tensor, record in ((before, initial["actual_parameter_hashes"]), (after, expected)):
            payload = ("torch." + str(tensor.dtype) + str(tuple(tensor.shape))).encode() + tensor.tobytes()
            if sha256(payload).hexdigest() != record[name]:
                raise ValueError("Saved parameter bytes differ from measured training parameters")
        changed += int(not np.array_equal(before, after))
        parameters += after.size
    if not changed:
        raise ValueError("Training produced identical adapter weights")
    return {"tensors": len(second), "tensors_changed": changed, "trainable_parameters": parameters,
            "parent_sha256": file_sha256(parent / "adapter/adapter_model.safetensors"),
            "child_sha256": file_sha256(child / "adapter/adapter_model.safetensors")}


def normalized_question(text):
    return re.findall(r"\w+", (text or "").casefold())


def score(parent, child, trial, out):
    protocol = json.loads((trial / "protocol.json").read_bytes())
    tensor = tensor_comparison(parent, child)
    targets = protocol["targets_scoring_only"]
    if file_sha256(trial / "requests.jsonl") != protocol["requests_sha256"]:
        raise ValueError("Evaluation inputs changed after freezing")
    results, pairs, settings = {}, {}, {}
    for cond, sha in (("public_parent", tensor["parent_sha256"]), ("human_child", tensor["child_sha256"])):
        folder = trial / cond
        native = check_native_run(folder)
        settings[cond] = json.loads((folder / "run.json").read_bytes())["configuration"]
        settings[cond].pop("adapter_dir")
        if any(settings[cond].get(k) != v for k, v in protocol["inference"].items()):
            raise ValueError("Native inference differs from the registered configuration")
        if (native["adapter_weights_sha256"] != sha or len(targets) != native["requested"]
                or native["files"]["input-snapshot.jsonl"] != protocol["requests_sha256"]):
            raise ValueError("Unexpected evaluation cohort or adapter")
        pairs[cond] = {r["request_id"]: (r["request_sha256"], r["image_sha256"]) for r in native["rows"]}
        if set(pairs[cond]) != set(targets):
            raise ValueError("Evaluation is missing a registered request")
        details = []
        for row in native["rows"]:
            ref = targets[row["request_id"]]
            result = row.get("result", {})
            expected = ref["target"]
            question = result.get("clarification_question")
            details.append({"request_id": row["request_id"], "reference": ref["group"], "ok": row["ok"],
                            "expected": expected["status"], "actual": result.get("status"), "question": question,
                            "correct_decision": result.get("status") == expected["status"],
                            "exact_reviewed_question": (normalized_question(question) ==
                                normalized_question(expected["clarification_question"]))
                                if expected.get("clarification_question") else None, "error": row.get("error")})
        results[cond] = {"rows": details, "operational_errors": native["operational_errors"],
                         "run_sha256": native["run_sha256"], "files": native["files"]}
    if pairs["public_parent"] != pairs["human_child"] or settings["public_parent"] != settings["human_child"]:
        raise ValueError("Conditions used different pixels, text or generation settings")
    training = json.loads((child / "training_result.json").read_bytes())
    if not training["actual_training_completed"] or training["injected_runtime"] or training["coverage"] != "clarification-only":
        raise ValueError("Expected an actual human-reviewed clarification run")
    save(out, {"kind": "actual_8B_human_confirmed_single_phrase_clarification_trial", "source_revision": REVISION,
               "protocol_sha256": file_sha256(trial / "protocol.json"), "training_result_sha256": file_sha256(child / "training_result.json"),
               "training": {k: training[k] for k in ("training_examples_unique", "examples_consumed", "optimizer_steps",
                               "epochs_completed", "parent_development_nll", "selected_checkpoint", "status_counts", "elapsed_s")},
               "tensor_verification": tensor, "fresh_generation": results,
               "independent_final_evaluation": False, "used_reserved_test_rows": 0,
               "scope": "Four training and two development photographs from six EMT recordings repeat one request "
                        "and one approved question. Reviewed language was AI-assisted and received two recorded confirmations. "
                        "The held-out photographs share the training wording. Neither spatial performance nor language "
                        "generalization is established by these two labels."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    for name in ("data", "control-requests", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    s = sub.add_parser("score")
    for name in ("parent", "child", "trial", "out"):
        s.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.data, args.control_requests, args.out)
    else:
        score(args.parent, args.child, args.trial, args.out)


if __name__ == "__main__":
    main()
