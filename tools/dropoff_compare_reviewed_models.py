"""Check the registered pretrained, public-adapted and locally continued runs on six nonfinal requests."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

from tools.dropoff_ai_acceptance import check_native_run
from tools.dropoff_reviewed_trial import normalized_question, save
from trackrag.dropoff.data import file_sha256
from trackrag.dropoff.formatting import build_messages


def verify_prompt(receipt, settings):
    """Verify the messages actually recorded at inference against model-visible inputs alone."""
    source = receipt["provenance"]
    messages = receipt.get("messages")
    if messages is None and receipt.get("status") == "error" and "messages_sha256" not in source:
        return None  # An error before prompt construction remains in the denominator.
    expected = build_messages(receipt["request"], allow_annotated=settings["allow_annotated"],
                              observation_format=settings["observation_format"])
    digest = sha256(json.dumps(expected, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if messages != expected or source.get("messages_sha256") != digest:
        raise ValueError("Recorded prompt differs from the original passenger request, photo or observations")
    return digest


def compare(trial, conditions, earlier_results):
    registration = json.loads(conditions.read_bytes())
    followup = registration["nonfinal_followup"]
    if (file_sha256(trial/"protocol.json") != followup["existing_six_case_scoring_protocol_sha256"]
            or file_sha256(trial/"requests.jsonl") != followup["requests_sha256"]):
        raise ValueError("Inputs differ from the registered six-case comparison")
    earlier = json.loads(earlier_results.read_bytes())
    if earlier["protocol_sha256"] != followup["existing_six_case_scoring_protocol_sha256"]:
        raise ValueError("The earlier adapter comparison used a different study")
    config = Path(registration["pretrained"]["config_path"])
    if file_sha256(config) != registration["pretrained"]["config_sha256"]:
        raise ValueError("The registered pretrained configuration changed")
    settings = json.loads(config.read_bytes())
    weights = {"pretrained": None, "public_parent": earlier["tensor_verification"]["parent_sha256"],
               "human_child": registration["trained"]["adapter_weights_sha256"]}
    fingerprint = {}
    verified = {}
    for name, expected in weights.items():
        run = trial/name
        native = check_native_run(run)
        cfg = json.loads((run/"run.json").read_bytes())["configuration"]
        if (native["requested"] != followup["requested_rows"]
                or native["files"]["input-snapshot.jsonl"] != followup["requests_sha256"]
                or native["adapter_weights_sha256"] != expected or (cfg | {"adapter_dir": None}) != settings):
            raise ValueError(f"{name} has a different request bank, weights or inference settings")
        if name in earlier["fresh_generation"] and native["run_sha256"] != earlier["fresh_generation"][name]["run_sha256"]:
            raise ValueError(f"The original {name} output has changed")
        receipts = [json.loads(line) for line in (run/"receipts.jsonl").read_bytes().splitlines() if line.strip()]
        for row, receipt in zip(native["rows"], receipts, strict=True):
            row["verified_prompt_sha256"] = verify_prompt(receipt, cfg)
        identities = {r["request_id"]: (r["request_sha256"], r["image_sha256"]) for r in native["rows"]}
        if fingerprint and identities != fingerprint:
            raise ValueError(f"{name} saw different words, photos or observations")
        fingerprint = identities
        verified[name] = native

    targets = json.loads((trial/"protocol.json").read_bytes())["targets_scoring_only"]
    if set(targets) != set(fingerprint):
        raise ValueError("Targets do not match the six verified requests")
    results = {}
    for name, native in verified.items():
        strata = defaultdict(lambda: {"requested": 0, "matching_status": 0, "operational_errors": 0})
        rows = []
        for row in native["rows"]:
            ref = targets[row["request_id"]]
            target = ref["target"]
            result = row.get("result", {})
            matches = result.get("status") == target["status"]
            counts = strata[ref["group"]]
            counts["requested"] += 1
            counts["matching_status"] += int(matches)
            counts["operational_errors"] += int(not row["ok"])
            rows.append({**row, "reference_group": ref["group"], "expected_status": target["status"],
                         "matching_status": matches, "exact_reviewed_question":
                             normalized_question(result.get("clarification_question")) ==
                             normalized_question(target["clarification_question"])
                             if target.get("clarification_question") else None})
        results[name] = {"run_sha256": native["run_sha256"], "raw_files": native["files"],
                         "weights_sha256": weights[name], "by_reference_group": dict(strata), "rows": rows}
    return {"kind": "three_matched_native_models_on_six_nonfinal_inputs",
            "registered_pair_sha256": file_sha256(conditions), "original_trial_sha256": earlier["protocol_sha256"],
            "model_outputs_checked": 18, "reserved_predictions": 0, "settings_excluding_adapter": settings,
            "conditions": results}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--trial", type=Path, required=True)
    cli.add_argument("--conditions", type=Path, required=True)
    cli.add_argument("--earlier-results", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    save(args.out, compare(args.trial, args.conditions, args.earlier_results))
