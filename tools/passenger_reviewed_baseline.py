"""Audit the signed, image-only Gulf requests and score two existing direct runs.

Both commands are readers. They never invoke a model or export individual
requests, references, reviewer identities, photographs or model responses.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from inspect import getsource
from io import StringIO
import json
from math import dist, hypot
from pathlib import Path

from evaluation.grounding import box_iou
from evaluation.grounding_inputs import TASK5_RUN_FILES
from tools.dropoff_final_input_inventory import FINAL, PACKAGE, ROOT, TASK2, git
from tools.dropoff_native_episodes import digest, moment, verify_native_receipt
from tools.dropoff_predict import read_requests
from trackrag.dropoff.contracts import FULL_FIELDS, validate_request, validate_target
from trackrag.dropoff.data import file_sha256, inference_input_sha256, supervision_sha256
from trackrag.dropoff.formatting import build_messages, parse_model_output
from trackrag.dropoff.inputs import _inspect_image
from trackrag.dropoff.model import BASE_MODEL_ID, BASE_REVISION, GroundingConfig, verify_adapter
from trackrag.dropoff.projected_pair import native_record


REQUEST_HASH = "fe8c8e310bc1c44eb8e768eb05b0f591c224c9baab85c838050d8954d777ad2c"
BINDING_HASH = "bf4aaf9132506e3eb1ff000e7b55db6c261be60d693b464e441a2c0387669b6b"
REGISTRY = ROOT / "experiments/dropoff/passenger-final-model-conditions-20261122.json"
REGISTRY_HASH = "a8483a2cee1d17b4b919c7c3e39f96f651dfe4f9178e524194a7518504543c67"
SOURCE_FILES = ("review_cases.jsonl", "reconfirmation_cases.jsonl", "reviewer_1.csv", "reviewer_2.csv")
STATUSES = ("proposed", "needs_clarification", "insufficient_visual_evidence")
FAILURES = ("invalid_model_output", "operational_error")
ERROR_CODES = {"artifact_invalid", "timeout", "dependency_unavailable", "resource_unavailable", "invalid_request",
               "invalid_image", "invalid_model_output", "checkpoint_unavailable", "generation_failed"}
SCORING_RULES = {
    "scope": "Direct typed, original image and unchanged empty observation list. Two fixed models, all 44 cases.",
    "primary_status": "Fraction matching the approved status, with every failed request counted as incorrect.",
    "point_in_signed_box": "Among all reviewed proposals, inclusive left/top/right/bottom box containment. Absent points score zero.",
    "point_distance": "Euclidean distance to the signed reference point divided by the original image diagonal. Report the mean over returned proposals and their coverage, plus an all-eligible mean charging one diagonal for each missing proposal or failure.",
    "box_overlap": "Original-pixel rectangle IoU averaged over every reviewed proposal. Missing proposals and failures contribute zero. Also report the number at IoU >= 0.5.",
    "clarification": "Literal case-sensitive Unicode equality with the approved question on all clarification-labelled cases; other outputs and failures score zero. This compares wording only.",
    "paired": "Counts of both correct, pretrained only, trained only and neither for matched statuses across every case.",
    "unit": "Each reviewed frame is one case; repeated instructions stay in their assigned recording. Metrics describe this cohort.",
    "interpretation": "Signed coarse boxes and points support geometric proxies. The study does not measure pixel-mask accuracy, passenger agreement with a clarification, destination confirmation, or the formal four-condition protocol.",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def json_rows(payload):
    return [json.loads(line) for line in payload.splitlines() if line.strip()]


def indexed(rows, key):
    values = {r[key]: r for r in rows}
    require(len(values) == len(rows), "Repeated row identity in a source or evidence file")
    return values


@dataclass
class Reference:
    request: dict
    target: dict
    image_sha256: str


def confirmed_reference(case, request, binding, forms, *, as_of=None):
    """Independently bind both signed CSV rows to the original input and four supervised fields."""
    target, review = case["target"], case["review"]
    old = validate_request(case["request"])
    request = validate_request(request)
    frame = request["scene"]["frame"]
    image_hash = case["source"]["image_sha256"]
    fields = case["supervised_fields"]
    require(case["partition"] == "test" and case["case_id"] == request["request_id"] == binding["case_id"],
            "Case, partition and reviewed input differ")
    old["scene"]["frame"]["image_path"] = frame["image_path"]
    require(old == request and request["input_mode"] == "typed" and request["dialogue"] == []
            and request["scene"]["source"] == "predicted" and request["scene"]["tracks"] == []
            and request["scene"]["provenance"] == {"mode": "image_only_local_annotation", "detector_run": False},
            "Use exactly the image-only request as reviewed")
    require(frame["recording_id"] == binding["recording_id"] == case["source"]["recording_id"]
            and frame["frame_id"] == binding["frame_id"] == case["source"]["frame_id"]
            and image_hash == review["image_sha256"] == binding["image_sha256"], "Unbound reviewed image")
    require(set(fields) == FULL_FIELDS and len(fields) == len(FULL_FIELDS), "Expected four signed target fields")
    validated = validate_target(target, fields, frame)
    input_hash = inference_input_sha256(request, image_hash)
    target_hash = supervision_sha256(validated, fields, frame)
    require(review["input_sha256"] == binding["reviewed_input_sha256"] == input_hash
            and review["supervision_sha256"] == target_hash and review["status"] == "human_reviewed"
            and review["new_human_review"] is True and review["independent_reviewers_required"] == 2,
            "Approved record has different input, supervision or review basis")
    require(len(forms) == 2 and len(set(review["reviewers"])) == 2
            and {f["reviewer_id"] for f in forms} == set(review["reviewers"]), "Two distinct reviewers are required")
    require(moment(review["reviewed_at"]) <= (as_of or datetime.now(timezone.utc)),
            "The source review postdates this study freeze")
    for form, stamp in zip(forms, (review["first_reviewed_at"], review["second_reviewed_at"]), strict=True):
        require(form["decision"] == "accept" and bool(form["reviewer_id"].strip())
                and form["reviewed_at"] == stamp and moment(stamp) <= moment(review["reviewed_at"])
                and form["case_id"] == case["case_id"] and form["partition"] == "test"
                and form["recording_id"] == frame["recording_id"] and int(form["frame_id"]) == frame["frame_id"]
                and form["image_sha256"] == image_hash and form["input_sha256"] == input_hash
                and form["supervision_sha256"] == target_hash and form["instruction"] == request["text"]
                and form["status"] == target["status"]
                and (form["clarification_question"] or None) == target["clarification_question"],
                "A reviewer did not attest the exact source, status and wording")
        for field in ("destination_point_xy", "destination_box_xyxy"):
            require((json.loads(form[field]) if form[field].strip() else None) == target[field],
                    "A reviewer did not attest the exact coordinates")
    require(_inspect_image(frame)[1] == image_hash, "Installed original photo hash differs")
    return Reference(request, validated, image_hash)


def reviewed_bank(path, as_of):
    payload = path.read_bytes()
    bindings_file = path.parent / "input-bindings.jsonl"
    require(sha256(payload).hexdigest() == REQUEST_HASH and file_sha256(bindings_file) == BINDING_HASH,
            "Expected the published original input snapshot and bindings")
    requests = read_requests(path, payload=payload)
    bindings = indexed(json_rows(bindings_file.read_bytes()), "case_id")
    sources = {name: git(TASK2, PACKAGE + "/" + name) for name in SOURCE_FILES}
    cases = json_rows(sources["review_cases.jsonl"])
    admitted = indexed([r for r in cases if r["partition"] == "test"], "case_id")
    pending = indexed(json_rows(sources["reconfirmation_cases.jsonl"]), "case_id")
    forms = [indexed(list(csv.DictReader(StringIO(sources[n].decode("utf-8-sig")))), "case_id")
             for n in ("reviewer_1.csv", "reviewer_2.csv")]
    require(len(admitted) == len(requests) == 44 and set(admitted) == set(bindings) == {r["request_id"] for r in requests}
            and len(pending) == 6 and not set(pending).intersection(admitted)
            and all(r["partition"] == "test" for r in pending.values()),
            "Original signed cohort, partitions or pending exclusions differ")
    require(all(all(not f[i]["decision"] for i in pending) for f in forms), "Pending cases require a new protocol")
    require({r["scene"]["frame"]["recording_id"] for r in requests} == FINAL, "Frozen recordings differ")
    refs = {}
    for q in requests:
        i = q["request_id"]
        refs[i] = confirmed_reference(admitted[i], q, bindings[i], [f[i] for f in forms], as_of=as_of)
    statuses = dict(sorted(Counter(ref.target["status"] for ref in refs.values()).items()))
    official = {
        "signed_inputs_matching_saved_finetuned_detector": sum(b["saved_detector_input_sha256"] == b["reviewed_input_sha256"] for b in bindings.values()),
        "zero_observation_scenes_per_saved_detector": {
            kind: sum(b["saved_detector_conditions"][kind]["current_observations"] == 0 for b in bindings.values())
            for kind in ("finetuned", "pretrained")},
        "source_records_with_destination_mask_path": sum(bool(c["source"].get("mask_path")) for c in admitted.values()),
        "original_prior_dialogues": sum(bool(r.request["dialogue"]) for r in refs.values()),
    }
    audit = {"approved_by_partition": dict(sorted(Counter(c["partition"] for c in cases).items())),
             "withheld_pending_test_cases": len(pending), "verified_test_cases": len(refs),
             "recordings": len(FINAL), "distinct_typed_requests": len({r.request["text"] for r in refs.values()}),
             "signed_status_counts": statuses,
             "distinct_signed_clarification_wordings": len({r.target["clarification_question"] for r in refs.values()
                                                            if r.target["clarification_question"] is not None}),
             "signed_point_and_box_pairs": sum(r.target["destination_point_xy"] is not None and
                                                r.target["destination_box_xyxy"] is not None for r in refs.values()),
             "source_checks": official, "review_basis": "Two reviewers accepted AI-assisted annotations on the original photographs and empty observed context.",
             "task2_commit": TASK2, "source_files_sha256": {n: sha256(b).hexdigest() for n, b in sources.items()},
             "request_snapshot_sha256": REQUEST_HASH, "release_bindings_sha256": BINDING_HASH}
    return refs, audit


def registered_models():
    require(file_sha256(REGISTRY) == REGISTRY_HASH, "Model-selection registry differs")
    registry = json.loads(REGISTRY.read_bytes())
    configs = {}
    for role in ("pretrained", "trained"):
        record = registry[role]
        path = ROOT / record["config_path"]
        require(file_sha256(path) == record["config_sha256"], "Installed configuration differs from the registered pair")
        configs[role] = GroundingConfig.from_mapping(json.loads(path.read_bytes()))
    before, after = configs["pretrained"], configs["trained"]
    require(before.adapter_dir is None and after.adapter_dir is not None and replace(after, adapter_dir=None) == before
            and before.runtime == "streamed" and before.local_files_only and not before.allow_annotated,
            "The registered direct configurations must differ only by the trained adapter")
    adapter = verify_adapter(replace(after, adapter_dir=str(ROOT / after.adapter_dir)))
    require(adapter["manifest_sha256"] == registry["trained"]["adapter_manifest_sha256"]
            and adapter["files"]["adapter_model.safetensors"]["sha256"] == registry["trained"]["adapter_weights_sha256"],
            "Installed adapter differs from the registered training artifact")
    return configs, registry


def eligibility(path, frozen_utc):
    frozen = moment(frozen_utc)
    require(frozen <= datetime.now(timezone.utc), "The study freeze must have occurred")
    refs, audit = reviewed_bank(path, frozen)
    configs, registry = registered_models()
    shared = (box_iou, read_requests, validate_request, validate_target, _inspect_image, inference_input_sha256,
              supervision_sha256, verify_adapter, native_record, verify_native_receipt, build_messages, parse_model_output)
    report = {"kind": "original_as_reviewed_gulf_direct_preanalysis", "predeclared_utc": frozen_utc,
              "source_audit": audit, "scoring_rules": SCORING_RULES, "model_registry_sha256": REGISTRY_HASH,
              "config_sha256": {role: registry[role]["config_sha256"] for role in configs},
              "trained_adapter_manifest_sha256": registry["trained"]["adapter_manifest_sha256"],
              "trained_adapter_weights_sha256": registry["trained"]["adapter_weights_sha256"],
              "analysis_scope": "Review counts and available supervision were audited before preregistration. The status, box, point and question metrics are fixed before collecting any new native responses for this comparison.",
              "scorer_sha256_lf": sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
              "shared_function_sha256_lf": {f"{f.__module__}.{f.__name__}": sha256(getsource(f).encode()).hexdigest()
                                            for f in shared},
              "eligibility_model_calls": 0}
    return report, refs, configs, registry


def verify_outcome(request, receipt, outcome, role, config, registry, image_hash):
    p = outcome["provenance"]
    require(receipt["request_id"] == outcome["request_id"] == request["request_id"] and receipt.get("request") == request
            and p["request_sha256"] == digest(request) and p["injected"] is False
            and p["backend_kind"] == "transformers_local_streamed" and p["requested_condition"] == ("fine_tuned" if role == "trained" else "pretrained")
            and p["revision"] == config.revision and p["model_id"] == config.model_id,
            "Native request, role or model source differs")
    for record in (p, outcome, receipt):
        require(all(record.get(k) in (False, None) for k in ("injected", "fixture", "software_fixture", "substitute", "test_fixture")),
                "Substitute evidence is ineligible")
    expected_generation = {key: getattr(config, key) for key in ("min_pixels", "max_pixels", "max_time_s", "max_new_tokens", "max_input_tokens")}
    require(p["generation"] == {"do_sample": False, **expected_generation}, "Generation settings differ")
    verify_native_receipt(request, outcome, receipt, config)
    image = p.get("image")
    require((image and image["sha256"] == image_hash) or (image is None and not outcome["ok"]),
            "Native input image differs from the reviewed original")
    adapter = p.get("adapter")
    if role == "pretrained":
        require(adapter is None, "Pretrained outcome has a trained adapter")
    elif outcome["ok"] or adapter is not None:
        require(adapter and adapter["manifest_sha256"] == registry["trained"]["adapter_manifest_sha256"]
                and adapter["files"]["adapter_model.safetensors"]["sha256"] == registry["trained"]["adapter_weights_sha256"],
                "Native trained weights differ from the registered condition")
    backend = p.get("backend_details")
    if outcome["ok"]:
        native_record(p, config)
        require(backend["processor_model_id"] == BASE_MODEL_ID and backend["processor_revision"] == BASE_REVISION
                and backend["effective_resident_layers"] == config.inference_resident_layers,
                "Processor or inference layout differs from the registered condition")
    else:
        # A worker startup, image, dependency or timeout failure can have incomplete
        # device/weight details. Preserve the failure without certifying computation.
        if backend:
            require(all(backend.get(k) in (None, False) for k in ("fixture", "software_fixture", "substitute", "injected", "test_fixture")),
                    "Substitute failure provenance is ineligible")
        if (outcome["error"]["code"] == "invalid_model_output" and isinstance(receipt.get("raw_text"), str)
                and "invalid_backend_details" not in receipt):
            frame = request["scene"]["frame"]
            try:
                parse_model_output(receipt["raw_text"], frame["width"], frame["height"])
            except ValueError:
                pass
            else:
                raise ValueError("Declared invalid raw response parsed as a valid native response")


def checked_run(folder, refs, role, config, registry, frozen_utc, launch):
    metadata = json.loads((folder / "run.json").read_bytes())
    run_hash = file_sha256(folder / "run.json")
    started, ended = moment(launch["started_utc"]), moment(launch["end_utc"])
    require(launch["run_sha256"] == run_hash and moment(frozen_utc) <= started <= moment(metadata["created_at"]) <= ended
            <= datetime.now(timezone.utc), "Launch seal has different run bytes or timing")
    require(metadata["configuration"] == asdict(config)
            and file_sha256(folder / "input-snapshot.jsonl") == metadata["requests_sha256"] == REQUEST_HASH,
            "Completed run has different settings or input bytes")
    files = {}
    for filename, key in TASK5_RUN_FILES:
        payload = (folder / filename).read_bytes()
        require(sha256(payload).hexdigest() == metadata[key], "Native run member differs from its sealed digest")
        files[filename] = json_rows(payload)
    expected = list(refs)
    for rows in files.values():
        require([r["request_id"] for r in rows] == expected, "Incomplete, repeated or reordered native cohort")
    outcomes = files["model-outcomes.jsonl"]
    succeeded = sum(o["ok"] is True for o in outcomes)
    require(metadata["requested_rows"] == len(expected) and metadata["semantic_results"] == succeeded
            and metadata["operational_errors"] == len(expected) - succeeded, "Native result denominator differs")
    for q, receipt, outcome in zip(files["resolved-requests.jsonl"], files["receipts.jsonl"], outcomes, strict=True):
        ref = refs[q["request_id"]]
        local = deepcopy(ref.request)
        # The archived Windows path may move. The actual receipt still must bind the exact recorded path.
        source_tail = local["scene"]["frame"]["image_path"].replace("\\", "/").split("/")[-5:]
        resolved_tail = q["scene"]["frame"]["image_path"].replace("\\", "/").split("/")[-5:]
        require(source_tail == resolved_tail, "Resolved photo path differs from the original release recording/frame")
        local["scene"]["frame"]["image_path"] = q["scene"]["frame"]["image_path"]
        require(q == local, "Resolved request differs from the portable reviewed source")
        verify_outcome(q, receipt, outcome, role, config, registry, ref.image_sha256)
    return indexed(outcomes, "request_id"), {"native_run_sha256": run_hash, "input_sha256": REQUEST_HASH,
                                             "started_utc": launch["started_utc"], "end_utc": launch["end_utc"],
                                             "native_file_sha256": {n: metadata[k] for n, k in TASK5_RUN_FILES}}


def mean(values):
    return sum(values) / len(values) if values else None


def model_metrics(refs, outcomes):
    require(set(refs) == set(outcomes), "Each reference requires one native outcome including failures")
    table = {s: dict.fromkeys((*STATUSES, *FAILURES), 0) for s in STATUSES}
    error_codes = Counter()
    distances, overlaps = [], []
    hits = eligible = exact = asked = 0
    correct = {}
    for case, ref in refs.items():
        expected = ref.target
        out = outcomes[case]
        result = out["result"] if out["ok"] else {}
        if out["ok"]:
            status = result["status"]
        else:
            code = out["error"]["code"]
            error_codes[code if code in ERROR_CODES else "other"] += 1
            status = "invalid_model_output" if code == "invalid_model_output" else "operational_error"
        table[expected["status"]][status] += 1
        correct[case] = status == expected["status"]
        if expected["status"] == "needs_clarification":
            asked += 1
            exact += status == "needs_clarification" and result["clarification_question"] == expected["clarification_question"]
        if expected["status"] == "proposed":
            eligible += 1
            if status == "proposed":
                point, box = result["destination_point_xy"], expected["destination_box_xyxy"]
                hits += box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]
                f = ref.request["scene"]["frame"]
                distances.append(dist(point, expected["destination_point_xy"]) / hypot(f["width"], f["height"]))
                overlaps.append(box_iou(result["destination_box_xyxy"], box))
            else:
                overlaps.append(0.0)
    n = len(refs)
    missing = eligible - len(distances)
    return {"cases": n, "valid_semantic_outputs": n - sum(error_codes.values()), "native_failed_rows": sum(error_codes.values()),
            "malformed_model_outputs": error_codes["invalid_model_output"],
            "operational_errors_excluding_malformed": sum(error_codes.values()) - error_codes["invalid_model_output"],
            "status_correct": sum(correct.values()), "status_accuracy": sum(correct.values()) / n if n else None,
            "status_confusion_expected_then_returned": table, "failed_requests_by_code": dict(sorted(error_codes.items())),
            "reference_proposals": eligible, "returned_proposals_on_reference_proposals": len(distances),
            "missing_proposals_or_failures_on_reference_proposals": missing,
            "points_in_signed_box": hits, "point_in_signed_box_rate_all_eligible": hits / eligible if eligible else None,
            "point_error_image_diagonals_mean_returned": mean(distances),
            "point_error_image_diagonals_mean_all_missing_cost_one": mean(distances + [1.] * missing),
            "signed_box_iou_mean_all_missing_zero": mean(overlaps), "signed_box_iou_at_least_half": sum(v >= .5 for v in overlaps),
            "reviewed_clarifications": asked, "clarification_question_literal_matches": exact,
            "clarification_question_literal_match_rate": exact / asked if asked else None}, correct


def score(path, seal, execution_file, folders):
    previously = json.loads(seal.read_bytes())
    frozen = previously["predeclared_utc"]
    report, refs, configs, registry = eligibility(path, frozen)
    require(report == previously, "Preanalysis record, sources or scoring implementation changed")
    launches = json.loads(execution_file.read_bytes())
    require(launches["eligibility_sha256"] == file_sha256(seal) and set(launches["conditions"]) == set(configs),
            "Execution seal must bind the exact preanalysis and both native conditions")
    models, matches = {}, {}
    for role in ("pretrained", "trained"):
        outcomes, evidence = checked_run(folders[role], refs, role, configs[role], registry, frozen,
                                         launches["conditions"][role])
        metrics, matches[role] = model_metrics(refs, outcomes)
        models[role] = {**metrics, **evidence}
    pairs = Counter((matches["pretrained"][i], matches["trained"][i]) for i in refs)
    return {"kind": "original_as_reviewed_gulf_direct_comparison", "preanalysis_sha256": file_sha256(seal),
            "execution_seal_sha256": file_sha256(execution_file),
            "preanalysis": report, "models": models,
            "paired_status_agreement": {name: pairs[key] for key, name in (((True, True), "both_correct"),
                ((True, False), "pretrained_only"), ((False, True), "trained_only"), ((False, False), "neither"))},
            "official_four_condition_score": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("eligibility", "score"):
        command = commands.add_parser(name)
        command.add_argument("--requests", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True, help="Aggregate-only JSON; existing evidence is preserved")
        if name == "score":
            for flag in ("eligibility", "execution", "pretrained", "trained"):
                command.add_argument("--" + flag, type=Path, required=True)
    a = parser.parse_args()
    if a.action == "eligibility":
        instant = (json.loads(a.out.read_bytes())["predeclared_utc"] if a.out.is_file() else datetime.now(timezone.utc).isoformat())
        data, *_ = eligibility(a.requests, instant)
    else:
        data = score(a.requests, a.eligibility, a.execution, {"pretrained": a.pretrained, "trained": a.trained})
    payload = (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if a.out.exists():
        require(a.out.read_bytes() == payload, "Existing report differs; retain the original")
    else:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open("xb") as stream:
            stream.write(payload)
    print(json.dumps({"kind": data["kind"], "aggregate_sha256": sha256(payload).hexdigest()}))


if __name__ == "__main__":
    main()
