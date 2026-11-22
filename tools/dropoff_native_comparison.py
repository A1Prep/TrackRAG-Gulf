"""Compare all three sealed native full-response Qwen runs against one fixed daytime bank."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from tools.dropoff_ai_acceptance import check_native_run
from tools.dropoff_passenger_references import score
from trackrag.dropoff.data import file_sha256


def read(path):
    return json.loads(path.read_bytes())


def compare(runs, profiles, requests, bank):
    executed = read(runs / "execution.json")
    original = read(profiles / "preflight.json")
    names = ("local_child", "public_parent", "pretrained")
    if (executed.get("state") != "completed" or not executed.get("end")
            or tuple(row["condition"] for row in executed["runs"]) != names):
        raise ValueError("Close all three GPU runs before comparing any results")
    request_digest = file_sha256(requests)
    if request_digest != original["requests_sha256"] or executed["inputs"] != 19:
        raise ValueError("The original matched bank changed")
    pinned = executed["source_sha256"]
    if any(file_sha256(Path(p)) != digest for p,digest in pinned.items()):
        raise ValueError("A pinned native file or profile no longer has its recorded bytes")
    reload_record = Path("experiments/dropoff/passenger-native-child-reload-20261121.json")
    prior_weights = read(reload_record)["cpu_independent_saved_tensor_verification"]
    weights = {"local_child": prior_weights["child_weights_sha256"],
               "public_parent": prior_weights["parent_weights_sha256"], "pretrained": None}
    profile_inputs = {r["request_id"]: r for r in original["rows"]}
    compared = {}
    reference_config = None
    for item in executed["runs"]:
        condition = item["condition"]
        folder = runs / condition
        cfg = read(profiles / (condition + ".json"))
        run_record = read(folder / "run.json")
        if run_record["configuration"] != cfg:
            raise ValueError("Native record differs from its frozen condition")
        base = {k:v for k,v in cfg.items() if k != "adapter_dir"}
        if reference_config is None:
            reference_config = base
        if base != reference_config or base["inference_resident_layers"] != 2 or base["max_time_s"] != 300:
            raise ValueError("Native conditions differ beyond the adapter")
        native = check_native_run(folder)
        if (native["requested"] != 19 or item["run_sha256"] != native["run_sha256"]
                or native["adapter_weights_sha256"] != weights[condition]
                or native["files"]["input-snapshot.jsonl"] != request_digest):
            raise ValueError("Missing rows, unexpected original adapter weights or mismatched native files")
        measured = score(bank, "development", requests, folder / "model-outcomes.jsonl")
        if measured["unexpected_output_ids"] or measured["requested"] != 19:
            raise ValueError("Unexpected or missing model identities")
        measurements = {r["request_id"]:r for r in measured.pop("per_case")}
        rows = []
        full_receipts = [json.loads(line) for line in (folder / "receipts.jsonl").read_bytes().splitlines()]
        for result, receipt in zip(native["rows"], full_receipts, strict=True):
            raw = result.get("raw_text")
            preflight = profile_inputs[result["request_id"]]
            source = receipt["provenance"]
            details = source.get("backend_details", {})
            message_sha = sha256(json.dumps(receipt["messages"], sort_keys=True, separators=(",", ":"),
                                             allow_nan=False).encode()).hexdigest()
            if (preflight["image_sha256"] != result["image_sha256"] or
                    message_sha != source["messages_sha256"] or message_sha != preflight["messages_sha256"]):
                raise ValueError("Generated input differs from the preflight image or native message")
            if (result["ok"] or raw) and (
                    details.get("input_tokens") != preflight["input_tokens"] or
                    not details.get("device", "").startswith("cuda") or
                    details.get("effective_resident_layers") != 2):
                raise ValueError("Genuine generation differs from the recorded GPU condition or token budget")
            record = {"request_id": result["request_id"],"ok":result["ok"],
                      "recorded_request_sha256":result["request_sha256"],"image_sha256":result["image_sha256"],
                      "recorded_message_sha256":message_sha,"actual_input_tokens":details.get("input_tokens"),
                      "raw_text_sha256":sha256(raw.encode()).hexdigest() if raw else None,
                      "native_four_field_result":result.get("result"),"error":result.get("error"),
                      "score":measurements.pop(result["request_id"])}
            rows.append(record)
        if measurements:
            raise ValueError("Unmatched native result rows")
        timing = {"worker_launched_utc": item["start"], "native_results_sealed_utc": run_record["created_at"],
                  "dispatcher_observed_exit_utc": item["end"], "exit_code": item["code"]}
        handoff_files = {"local_child": ("gpu-ready.json", "a separate training job"),
                         "public_parent": ("browser-gpu-ready.json", "separate passenger application checks")}
        if condition in handoff_files and (runs / handoff_files[condition][0]).is_file():
            filename, scheduled_job = handoff_files[condition]
            handoff = read(runs / filename)
            if (handoff["requested"] != native["requested"]
                    or handoff["operational_errors"] != native["operational_errors"]
                    or not any(p["pid"] == item["pid"] for p in handoff["verified_exited"])):
                raise ValueError("Independent worker-exit handoff disagrees with the native run")
            timing["independently_observed_process_tree_exit_utc"] = handoff["ready"]
            timing["worker_handoff_file_sha256"] = file_sha256(runs / filename)
            timing["timing_note"] = f"The dispatcher reserved time for {scheduled_job} after the native results sealed."
        compared[condition] = {"configuration_sha256":file_sha256(profiles / (condition + ".json")),
            "native_run_sha256":native["run_sha256"],"native_file_hashes":native["files"],
            "actual_weight_sha256":native["adapter_weights_sha256"],"operational_errors":native["operational_errors"],
            "native_process": timing,"metrics":measured,"cases":rows}
    outcomes={condition:{r["request_id"]:r for r in value["cases"]} for condition,value in compared.items()}
    comparisons={}
    for first,other in [("local_child","public_parent"),("local_child","pretrained"),("public_parent","pretrained")]:
        ordered=outcomes[first]; controls=outcomes[other]
        if ordered.keys() != controls.keys():
            raise ValueError("A native condition is missing an original case")
        status=Counter();strict=Counter();different=0
        for identity,a in ordered.items():
            b=controls[identity]
            if a["recorded_request_sha256"]!=b["recorded_request_sha256"] or a["image_sha256"]!=b["image_sha256"]:
                raise ValueError("Paired conditions consumed different model-visible inputs")
            ac,bc=a['score'],b['score']
            status[f"{int(ac['status_matches'])}->{int(bc['status_matches'])}"]+=1
            if ac['expected_status']=='proposed':
                strict[f"{int(ac['point_in_authored_box'])}->{int(bc['point_in_authored_box'])}"]+=1
            different += a['native_four_field_result']!=b['native_four_field_result'] or a['error']!=b['error']
        comparisons[first+"_vs_"+other]={"row_values_differ":different,
                "status_match_paired_left_to_right":dict(status),"strict_hits_on_clear_paired_left_to_right":dict(strict)}
    return {"kind":"matched_three_way_native_Qwen_full_response","measured_utc":datetime.now(timezone.utc).isoformat(),
            "source":"Real local streamed Qwen3-VL-8B four-field inference. Each source row contains the same original EMT pixels, predicted tracks and passenger dialogue.",
            "scope":"19 development instructions from one daytime recording. Their provisional references were AI-authored with a second blinded AI review. No human passenger labels, separate recordings or independent vehicle safety assessment.",
            "request_bank_sha256":request_digest,"frozen_profile_sha256":file_sha256(profiles/"preflight.json"),
            "verified_preexisting_adapter_reload_sha256": file_sha256(reload_record),
            "execution_sha256":file_sha256(runs/"execution.json"),"comparison_settings":reference_config,
            "source_hashes_frozen_for_execution":{k:v for k,v in pinned.items() if k.endswith(".py")},
            "conditions":compared,"paired":comparisons,"genuine_native_attempts":57}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ["runs","profiles","requests","bank","out"]:
        parser.add_argument("--"+n,type=Path,required=True)
    a=parser.parse_args()
    result=compare(a.runs,a.profiles,a.requests,a.bank)
    with a.out.open("x",encoding="utf8",newline="\n") as file:
        json.dump(result,file,indent=2,allow_nan=False)
        file.write("\n")
    print(json.dumps({k:{"status":r["metrics"]["status_matches"],"strict_point_hits":r["metrics"]["point_in_authored_box_count"],
                         "errors":r["operational_errors"]} for k,r in result["conditions"].items()}))


if __name__ == "__main__":
    main()
