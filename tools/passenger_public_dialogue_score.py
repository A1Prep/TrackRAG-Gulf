"""Score the frozen public concise/full instruction comparison after all model responses are sealed."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from statistics import mean
import tarfile

import numpy as np
from PIL import Image

from evaluation import grounding
from evaluation.grounding_inputs import TASK5_RUN_FILES
from tools import dropoff_bootstrap_regseg as dataset
from tools.dropoff_native_episodes import moment
from tools.passenger_reviewed_baseline import registered_models, verify_outcome
from trackrag.dropoff.contracts import validate_request
from trackrag.dropoff.data import file_sha256, inference_input_sha256


PROTOCOL = "8a858da1901c9a1c445010dad67db64fbb7d4fcbf5f78ea6e32391fe15fcb641"
COHORT = "5c51369808fe3f7ac83f7469a005ef8629cecee84fce4b97cada6234c8446961"
MASK_TAR = "c2f83477a50763cdcef9885bc20dd3ffc7202871c6f1fca4c268b6cceefc9d3f"
MODELS = ("pretrained", "trained")
STAGES = {f"{model}-turn{turn}" for model in MODELS for turn in (0, 1)}
ROOT = Path(__file__).resolve().parents[1]


def require(condition, explanation):
    if not condition:
        raise ValueError(explanation)


def read(path):
    return json.loads(path.read_bytes())


def rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def check_stage_order(order, execution):
    previous = moment(execution["started_utc"])
    for name in order:
        stage = execution["stages"][name]
        if stage["run_sha256"] is not None:
            start, end = moment(stage["started_utc"]), moment(stage["finished_utc"])
            require(previous <= start <= end <= moment(execution["completed_utc"]),
                    "Native stages were overlapping or outside their registered order")
            previous = end


def checked_seal(work):
    """Do all input/output identity checks before the referee opens any mask."""
    seal = read(work / "sealed-outputs.json")
    require(file_sha256(work / "protocol.json") == seal["protocol_sha256"] == PROTOCOL
            and file_sha256(work / "manifest.json") == seal["manifest_sha256"] == COHORT,
            "The protocol or source cohort differs from the registered experiment")
    protocol, cohort, execution = (read(work / name) for name in ("protocol.json", "manifest.json", "execution.json"))
    require(file_sha256(work / "selection.json") == cohort["selection_sha256"] == protocol["selection_sha256"]
            and file_sha256(work / "fetch_public.py") == protocol["source_selector_sha256"]
            and file_sha256(work / "study.py") == protocol["execution_source_sha256"],
            "Public source selection or inference code differs from the preregistration")
    for name, expected in protocol["source_hashes_lf"].items():
        content = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        require(sha256(content).hexdigest() == expected, f"Native model source differs: {name}")
    require(file_sha256(work / "execution.json") == seal["execution_sha256"]
            and execution["complete"] and execution["protocol_sha256"] == PROTOCOL
            and execution["manifest_sha256"] == protocol["manifest_sha256"] == COHORT
            and moment(protocol["frozen_utc"]) <= moment(execution["started_utc"])
            <= moment(execution["completed_utc"]) <= moment(seal["sealed_utc"]) <= datetime.now(timezone.utc)
            and set(seal["run_sha256"]) == set(execution["stages"]) == STAGES,
            "Missing, incomplete or out-of-order execution")
    require(cohort["target_pixels_read"] is False and seal["target_pixels_opened_by_runner"] is False
            and seal["passenger_confirmations"] == 0, "The run used information outside the registered study")
    check_stage_order(protocol["stage_order"], execution)
    cases = cohort["cases"]
    ids = [case["case_id"] for case in cases]
    require(ids == protocol["case_ids"] and len(set(ids)) == len(cases) == cohort["cohort_size"] == seal["registered_cases"]
            and len({case["source_scene"] for case in cases}) == len(cases),
            "Each registered input must carry its own case and source scene")
    initial_path, wording_path = work / "inputs/requests-first.jsonl", work / "inputs/source-language.jsonl"
    require(file_sha256(initial_path) == cohort["request_file_sha256"] == protocol["input_snapshot_sha256"]
            and file_sha256(wording_path) == cohort["language_file_sha256"] == protocol["language_bank_sha256"],
            "The published source language changed")
    initial, wording = rows(initial_path), rows(wording_path)
    for case, q, language in zip(cases, initial, wording, strict=True):
        q = validate_request(q)
        frame = q["scene"]["frame"]
        require(q["request_id"] == language["case_id"] == case["case_id"] and q["text"] == language["concise"]
                and [frame["width"], frame["height"]] == case["original_dimensions"]
                and sha256(q["text"].encode()).hexdigest() == case["first_sha256"]
                and sha256(language["full"].encode()).hexdigest() == case["followup_sha256"]
                and inference_input_sha256(q, case["image_sha256"]) == case["first_input_sha256"]
                and q["dialogue"] == [] and q["scene"]["tracks"] == [], "A source case, prompt or image changed")
    return cohort, seal, execution, dict(zip(ids, initial, strict=True)), dict(zip(ids, wording, strict=True))


def restatement(first, full, raw):
    next_request = deepcopy(first)
    frame = first["scene"]["frame"]
    same_frame = {"recording_id": frame["recording_id"], "frame_id": frame["frame_id"]}
    next_request["dialogue"] = [
        {"role": "user", "text": first["text"], **same_frame},
        {"role": "assistant", "text": raw, **same_frame},
    ]
    next_request["text"] = full
    return next_request


def verify_stage(work, name, expected, photos, seal, execution, config, registry):
    role = name.split("-")[0]
    stage = execution["stages"][name]
    source_file = ("requests-first.jsonl" if name.endswith("-turn0") else f"requests-{role}-followup.jsonl")
    inputs = work / "inputs" / source_file
    require(file_sha256(inputs) == stage["requests_sha256"], "Native run was launched on a different request bank")
    if not expected:
        require(name.endswith("-turn1") and stage["rows"] == 0 and seal["run_sha256"][name] is None
                and stage["reason"] == "zero eligible first native semantic responses"
                and inputs.read_bytes() == b"" and not (work / name).exists(),
                "A follow-up stage with zero eligible requests needs an explicit record")
        return {}
    folder = work / name
    summary = read(folder / "run.json")
    require(file_sha256(folder / "run.json") == seal["run_sha256"][name] == stage["run_sha256"]
            and stage["process_exit"] == 0 and asdict(config) == summary["configuration"]
            and registry[role]["config_sha256"] == stage["config_sha256"]
            and moment(execution["started_utc"]) <= moment(stage["started_utc"]) <= moment(summary["created_at"])
            <= moment(stage["finished_utc"]) <= moment(execution["completed_utc"]),
            "Native run, settings, model or execution times differ")
    contents = {}
    for filename, key in (("input-snapshot.jsonl", "requests_sha256"), *TASK5_RUN_FILES):
        require(file_sha256(folder / filename) == summary[key], f"Changed native artifact: {name}/{filename}")
        source = rows(folder / filename)
        require([r["request_id"] for r in source] == list(expected), f"Missing or reordered cases: {name}/{filename}")
        contents[filename] = source
    require(stage["requests_sha256"] == summary["requests_sha256"], "The launched batch and saved request snapshots differ")
    journal_path = folder / "per-call-journal.jsonl"
    require(file_sha256(journal_path) == stage["journal_sha256"], "The native per-call journal changed")
    journal = rows(journal_path)
    count = sum(row["ok"] is True for row in contents["model-outcomes.jsonl"])
    require(len(expected) == stage["rows"] == summary["requested_rows"]
            and count == summary["semantic_results"] and len(expected) - count == summary["operational_errors"],
            "A failed request disappeared from the native run")
    outcomes = {}
    previous = moment(stage["started_utc"])
    for source, q, receipt, outcome, entry in zip(contents["input-snapshot.jsonl"], contents["resolved-requests.jsonl"],
                                                  contents["receipts.jsonl"], contents["model-outcomes.jsonl"], journal, strict=True):
        case = q["request_id"]
        require(entry["request_id"] == case and entry["outcome"] == outcome and entry["receipt"] == receipt
                and previous <= moment(entry["started_utc"]) <= moment(entry["finished_utc"]) <= moment(summary["created_at"]),
                "A per-call journal entry differs from its natural execution order or raw result")
        previous = moment(entry["finished_utc"])
        require(source == expected[case], "Native snapshot differs from the exact registered dialogue")
        resolved = deepcopy(source)
        require(Path(resolved["scene"]["frame"]["image_path"]).name ==
                Path(q["scene"]["frame"]["image_path"].replace("\\", "/")).name, "The original photo filename changed")
        resolved["scene"]["frame"]["image_path"] = q["scene"]["frame"]["image_path"]
        require(q == resolved, "Native source request changed when its image path was resolved")
        verify_outcome(q, receipt, outcome, role, config, registry, photos[case])
        outcomes[case] = (outcome, receipt)
    return outcomes


def verified_runs(work):
    cohort, seal, execution, first_requests, wording = checked_seal(work)
    configs, registry = registered_models()
    photos = {case["case_id"]: case["image_sha256"] for case in cohort["cases"]}
    results = {}
    for role, config in configs.items():
        initial = verify_stage(work, role + "-turn0", first_requests, photos, seal, execution, config, registry)
        full = {
            cid: restatement(first_requests[cid], wording[cid]["full"], receipt["raw_text"])
            for cid, (outcome, receipt) in initial.items() if outcome["ok"]
        }
        second = verify_stage(work, role + "-turn1", full, photos, seal, execution, config, registry)
        results[role] = (initial, second)
    require(sum(len(stage) for pair in results.values() for stage in pair) == seal["actual_unique_gpu_calls"],
            "The total number of genuine calls differs from the seal")
    return cohort, seal, results


def read_annotation_archive(path):
    if path is not None:
        require(file_sha256(path) == MASK_TAR, "The public annotation archive differs from the registered source")
        return tarfile.open(path, "r:")
    source = dataset.BoundedSource(max_bytes=32 * 1024**2)
    tail = source.fetch(dataset.ARCHIVE_URL, dataset.ARCHIVE_SIZE - 65557, dataset.ARCHIVE_SIZE - 1,
                        total=dataset.ARCHIVE_SIZE)
    name = "Talk2Car-RefSeg/masks.tar"
    directory = dataset.zip_directory(tail, dataset.ARCHIVE_SIZE)
    data, receipt = dataset.read_outer_member(source, name, directory[name])
    require(receipt["complete"] and receipt["compressed_sha256"] == dataset.MASK_MEMBER_COMPRESSED_SHA256
            and sha256(data).hexdigest() == MASK_TAR, "Downloaded annotation archive differs")
    return tarfile.open(fileobj=BytesIO(data), mode="r:")


def target_masks(cohort, path):
    require(cohort["mask_tar_sha256_to_verify_after_seal"] == MASK_TAR, "The registered mask source changed")
    masks = {}
    with read_annotation_archive(path) as archive:
        for case in cohort["cases"]:
            index, name = case["source_index"], case["mask_member"]
            require(case["case_id"] == f"public-concise-{index}" and name in {
                f"masks/gt_img_ann_train_{index}.png", f"val_masks_new/gt_img_ann_train_{index}.png"
            }, "Mask and public command ID differ")
            try:
                member = archive.getmember(name)
            except KeyError:
                masks[case["case_id"]] = (None, None)
                continue
            if not member.isfile() or not 0 < member.size <= 4 * 1024**2:
                masks[case["case_id"]] = (None, None)
                continue
            with archive.extractfile(member) as stream:
                data = stream.read()
            digest = sha256(data).hexdigest()
            try:
                with Image.open(BytesIO(data)) as image:
                    require(image.format == "PNG" and 0 < image.width * image.height <= 20_000_000,
                            "Malformed source annotation")
                    mask = np.asarray(image.convert("L")) != 0
                if not mask.any():
                    mask = None
            except (OSError, ValueError, Image.DecompressionBombError):
                mask = None
            masks[case["case_id"]] = (mask, digest)
    return masks


def measure(outcome, mask, width, height):
    call, result, invalid = grounding.from_task5_outcome(outcome)
    if call["outcome"] == "error":
        status = "operational_error"
    else:
        status = grounding._classify(result, width, height, {"reasons": []}, invalid)
    hit, iou = False, 0.0
    if status == "proposed" and mask is not None:
        hit = grounding.in_mask(mask, result["point"], width, height)
        iou = grounding.box_iou(result["box"], grounding.mask_box(mask, width, height))
    return {"status": status, "point_in_mask": hit, "box_iou_with_mask_extent": iou}


def aggregate(row_values):
    rows_ = list(row_values)
    n = len(rows_)
    available = [r for r in rows_ if r["reference_available"]]
    hits = sum(r["point_in_mask"] for r in rows_)
    return {"cases": n, "usable_masks": len(available), "statuses": dict(sorted(Counter(r["status"] for r in rows_).items())),
            "point_in_mask_hits": hits, "point_in_mask_rate": hits / n,
            "point_in_mask_wilson95": grounding.wilson(hits, n),
            "mean_box_iou_with_mask_extent_missing_zero": mean(r["box_iou_with_mask_extent"] for r in rows_),
            "point_in_mask_rate_available_masks": hits / len(available) if available else None,
            "mean_box_iou_available_masks": mean(r["box_iou_with_mask_extent"] for r in available) if available else None}


def paired(before, after):
    require(before.keys() == after.keys(), "A paired comparison needs exactly the same source scenes")
    cells = Counter((before[cid]["point_in_mask"], after[cid]["point_in_mask"]) for cid in before)
    lost, gained = cells[True, False], cells[False, True]
    return {"both_hit": cells[True, True], "both_missed": cells[False, False], "gained": gained, "lost": lost,
            "point_hit_rate_difference": (gained - lost) / len(before),
            "two_sided_exact_mcnemar": grounding.mcnemar_exact(lost, gained)}


def score(work, mask_tar):
    cohort, seal, results = verified_runs(work)
    masks = target_masks(cohort, mask_tar)
    arm_rows, arms = {}, {}
    for model, (initial, follow) in results.items():
        direct, revised = {}, {}
        elapsed = {"direct": [], "restatement": []}
        for case in cohort["cases"]:
            cid = case["case_id"]
            mask, _ = masks[cid]
            outcome, _ = initial[cid]
            baseline = measure(outcome, mask, *case["original_dimensions"])
            baseline["reference_available"] = mask is not None
            direct[cid] = baseline
            duration = outcome["provenance"]["elapsed_s"]
            elapsed["direct"].append(duration)
            if cid in follow:
                outcome, _ = follow[cid]
                result = measure(outcome, mask, *case["original_dimensions"])
                result["reference_available"] = mask is not None
                revised[cid] = result
                duration += outcome["provenance"]["elapsed_s"]
            else:
                revised[cid] = {**baseline, "status": "ineligible_first_" + baseline["status"]}
            elapsed["restatement"].append(duration)
        for mode, values in (("direct", direct), ("restatement", revised)):
            arm_rows[model + "_" + mode] = values
            arms[model + "_" + mode] = {**aggregate(values.values()), "initial_calls": len(initial),
                "additional_calls": 0 if mode == "direct" else len(follow),
                "mean_total_native_elapsed_s_per_case": mean(elapsed[mode])}
    compared = {}
    for before, after in (("pretrained_direct", "trained_direct"), ("pretrained_restatement", "trained_restatement"),
                          ("pretrained_direct", "pretrained_restatement"), ("trained_direct", "trained_restatement")):
        compared[before + "_to_" + after] = paired(arm_rows[before], arm_rows[after])
    return {"kind": "paired_public_navigable_region_language_comparison", "protocol_sha256": PROTOCOL,
            "manifest_sha256": COHORT, "execution_sha256": seal["execution_sha256"], "run_sha256": seal["run_sha256"],
            "outputs_seal_sha256": file_sha256(work / "sealed-outputs.json"), "mask_archive_sha256": MASK_TAR,
            "actual_unique_model_calls": seal["actual_unique_gpu_calls"], "arms": arms, "paired": compared,
            "cases": [{"case_id": c["case_id"], "scene": c["source_scene"], "mask_sha256": masks[c["case_id"]][1],
                       "results": {arm: data[c["case_id"]] for arm, data in arm_rows.items()}} for c in cohort["cases"]],
            "passenger_confirmations": 0,
            "scope": "Twelve public source scenes with published concise and full commands and human-drawn regions. "
                     "The full command was supplied once after each successful first response. Commands include driving "
                     "and navigation as well as stopping. This measures a scripted restatement and its added compute."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True, help="Directory with registered inputs and all four native runs")
    masks = parser.add_mutually_exclusive_group(required=True)
    masks.add_argument("--mask-tar", type=Path, help="Original Talk2Car-RegSeg masks.tar")
    masks.add_argument("--fetch-masks", action="store_true", help="Read the verified mask member from the public source")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = score(args.work, args.mask_tar)
    data = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if args.out.exists():
        require(args.out.read_bytes() == data, "Existing report differs from the recomputed results")
    else:
        with args.out.open("xb") as stream:
            stream.write(data)
    print(json.dumps(result["arms"], indent=2))


if __name__ == "__main__":
    main()
