"""Independently score saved development trials; never run a model or alter outputs.

Public masks supervise destination geometry only. Missing, invalid, failed and
unresolved requests stay in the denominator; semantic status accuracy is unknown.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

from PIL import Image

from trackrag.dropoff.contracts import FULL_FIELDS, validate_target
from trackrag.dropoff.data import inference_input_sha256, load_cases
from trackrag.dropoff.formatting import parse_model_output
from tools.dropoff_method_trial import decode_selection, parse_selection, validate_candidates
from tools.dropoff_predict import read_requests


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate saved field: " + key)
        result[key] = value
    return result


def _finite(value):
    raise ValueError("Non-finite saved JSON constant: " + value)


def read_json(payload):
    return json.loads(payload, object_pairs_hook=_unique, parse_constant=_finite)


def read_lines(payload):
    return [read_json(line) for line in payload.decode("utf-8-sig").splitlines() if line.strip()]


def point_hit(point, mask, width, height):
    """Map original image pixels to source mask cells, without interpolation."""
    if (not isinstance(point, (list, tuple)) or len(point) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in point)):
        raise ValueError("Invalid original-image point")
    x, y = point
    if not 0 <= x < width or not 0 <= y < height:
        raise ValueError("Point outside original image")
    return mask.getpixel((math.floor(x * mask.width / width), math.floor(y * mask.height / height))) != 0


def _summarize(rows, coverable):
    n = len(rows)
    latencies = [r["elapsed_s"] for r in rows if r["elapsed_s"] is not None]
    hits = sum(r["point_in_original_mask"] for r in rows)
    covered_rows = [r for r in rows if r["case_id"] in coverable]
    return {
        "requested": n,
        "point_hits": hits,
        "point_agreement_all_requests": hits / n,
        "proposals": sum(r["status"] == "proposed" and r["operational_ok"] for r in rows),
        "proposal_coverage": sum(r["status"] == "proposed" and r["operational_ok"] for r in rows) / n,
        "operational_errors": sum(not r["operational_ok"] for r in rows),
        "missing_outputs": sum(r["error_type"] == "missing_output" for r in rows),
        "status_counts": dict(Counter(r["status"] for r in rows if r["operational_ok"])),
        "error_types": dict(Counter(r["error_type"] for r in rows if r["error_type"])),
        "mean_latency_s_observed_only": statistics.mean(latencies) if latencies else None,
        "median_latency_s_observed_only": statistics.median(latencies) if latencies else None,
        "total_latency_s_observed_only": sum(latencies),
        "latency_observations": len(latencies),
        "candidate_coverable_cases": len(covered_rows),
        "point_hits_among_candidate_coverable": sum(r["point_in_original_mask"] for r in covered_rows),
        "point_agreement_among_candidate_coverable": (sum(r["point_in_original_mask"] for r in covered_rows) / len(covered_rows)) if covered_rows else None,
        "semantic_status_accuracy": None,
        "cases": rows,
    }


def score(cases_path, protocol_path, candidates_path, outcomes_path):
    paths = {"development_cases": Path(cases_path), "protocol": Path(protocol_path),
             "candidate_manifest": Path(candidates_path), "outcomes": Path(outcomes_path)}
    paths.update(input_snapshot=paths["outcomes"].with_name("input-snapshot.jsonl"),
                 candidate_snapshot=paths["outcomes"].with_name("candidates-snapshot.jsonl"),
                 run_manifest=paths["outcomes"].with_name("run.json"))
    optional_paths = {"settings": paths["outcomes"].with_name("settings.json"),
                      "configuration": paths["outcomes"].with_name("configuration.json")}
    run_source = paths["outcomes"].with_name("inference-source-snapshot.py")
    if not run_source.exists():
        legacy_source = paths["outcomes"].parent.parent / "local-tool-start-snapshot.py"
        initial_settings = read_json(optional_paths["settings"].read_bytes()) if optional_paths["settings"].exists() else {}
        declared_source = initial_settings.get("source_file_sha256")
        if legacy_source.exists() and (not declared_source or sha(legacy_source.read_bytes()) == declared_source):
            run_source = legacy_source
    optional_paths["inference_source_snapshot"] = run_source
    for label, path in optional_paths.items():
        if path.exists():
            paths[label] = path
    snapshots = {key: path.read_bytes() for key, path in paths.items()}
    protocol = read_json(snapshots["protocol"])
    run = read_json(snapshots["run_manifest"])
    if run.get("outcomes_sha256") != sha(snapshots["outcomes"]):
        raise ValueError("Run manifest does not bind these outcome bytes; run may still be active")
    if protocol.get("requests_sha256") != sha(snapshots["input_snapshot"]):
        raise ValueError("Input snapshot differs from frozen protocol requests")
    if snapshots["candidate_manifest"] != snapshots["candidate_snapshot"]:
        raise ValueError("Candidate manifest differs from the inference snapshot")
    settings = read_json(snapshots["settings"]) if "settings" in snapshots else {}
    if settings.get("candidate_manifest_sha256") and settings["candidate_manifest_sha256"] != sha(snapshots["candidate_manifest"]):
        raise ValueError("Candidate manifest differs from declared inference settings")
    if (settings.get("source_file_sha256") and "inference_source_snapshot" in snapshots
            and settings["source_file_sha256"] != sha(snapshots["inference_source_snapshot"])):
        raise ValueError("Inference source snapshot differs from declared settings")
    ids = protocol["public_case_ids"]
    if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
        raise ValueError("Protocol must declare a nonempty unique public cohort")
    if protocol.get("development_source_sha256") != sha(snapshots["development_cases"]):
        raise ValueError("Development source differs from frozen protocol")
    loaded = load_cases(paths["development_cases"], allowed_partitions=("development",),
                        payload=snapshots["development_cases"])
    by_id = {r["case_id"]: r for r in loaded}
    if not set(ids) <= by_id.keys():
        raise ValueError("Protocol cohort missing from public development source")
    requests = {r["request_id"]: r for r in read_requests(paths["input_snapshot"], payload=snapshots["input_snapshot"])}
    for identity in ids:
        if identity not in requests:
            raise ValueError("Frozen input snapshot lacks a public case")
        case = by_id[identity]
        if inference_input_sha256(requests[identity], case["source"]["image_sha256"]) != inference_input_sha256(case["request"], case["source"]["image_sha256"]):
            raise ValueError("Frozen request differs from reference instruction/context: " + identity)
    options = {}
    for row in read_lines(snapshots["candidate_manifest"]):
        key = row["request_id"]
        if key in options:
            raise ValueError("Duplicate candidate request")
        options[key] = row
    outcomes = {}
    for row in read_lines(snapshots["outcomes"]):
        if row.get("provider") != "local" or row.get("request_id") not in ids:
            continue
        key = (row["request_id"], row.get("method"))
        if key[1] not in ("direct", "candidates") or key in outcomes:
            raise ValueError("Duplicate or unknown method in public outputs")
        outcomes[key] = row

    ceiling_rows, by_method = [], {"direct": [], "candidates": []}
    for case_id in ids:
        case = by_id[case_id]
        frame, source = case["request"]["scene"]["frame"], case["source"]
        mask_path = Path(source["mask_path"])
        mask_bytes = mask_path.read_bytes()
        if sha(mask_bytes) != source["mask_sha256"]:
            raise ValueError("Original mask hash differs: " + case_id)
        with Image.open(mask_path) as image:
            mask = image.convert("L")
        if case_id not in options:
            raise ValueError("Missing frozen candidates: " + case_id)
        candidates = validate_candidates(options[case_id], frame, source["image_sha256"])
        valid_ids = [c["candidate_id"] for c in candidates
                     if point_hit(c["point_original_px"], mask, frame["width"], frame["height"])]
        ceiling_rows.append({"case_id": case_id, "candidate_count": len(candidates),
                             "mask_sha256": sha(mask_bytes), "image_sha256": source["image_sha256"],
                             "representative_point_covered": bool(valid_ids), "mask_hitting_candidate_ids": valid_ids})
        for method in by_method:
            raw = outcomes.get((case_id, method))
            row = {"case_id": case_id, "operational_ok": False, "status": None,
                   "point_in_original_mask": False, "elapsed_s": None, "error_type": None}
            if raw is None:
                row["error_type"] = "missing_output"
            else:
                row["saved_output_sha256"] = sha(json.dumps(raw, sort_keys=True, allow_nan=False).encode())
                elapsed = raw.get("elapsed_s")
                if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and math.isfinite(elapsed) and elapsed >= 0:
                    row["elapsed_s"] = elapsed
                try:
                    if raw.get("ok") is not True:
                        error = raw.get("error")
                        row["error_type"] = error.get("type", "saved_request_error") if isinstance(error, dict) else "saved_request_error"
                        if not isinstance(row["error_type"], str) or not row["error_type"]:
                            row["error_type"] = "saved_request_error"
                    else:
                        if raw.get("request_sha256") != sha(json.dumps(requests[case_id], sort_keys=True).encode()):
                            raise ValueError("Saved request digest differs from frozen inference input")
                        if raw.get("original_image_sha256") != source["image_sha256"]:
                            raise ValueError("Output image differs from reference image")
                        messages = raw.get("messages")
                        if (not isinstance(messages, list) or not messages
                                or not isinstance(messages[-1], dict)
                                or messages[-1].get("role") != "user"
                                or messages[-1].get("content") != case["request"]["text"]):
                            raise ValueError("Saved model instruction differs from the frozen case")
                        if raw.get("messages_sha256") != sha(json.dumps(messages, sort_keys=True).encode()):
                            raise ValueError("Saved message digest differs from messages")
                        if not isinstance(raw.get("raw_text"), str) or len(raw["raw_text"]) > 32768:
                            raise ValueError("Raw output must be bounded text")
                        if method == "direct":
                            parsed = parse_model_output(raw["raw_text"], frame["width"], frame["height"])
                        else:
                            selection = parse_selection(raw["raw_text"])
                            parsed = decode_selection(selection, candidates, frame)
                            if "selection" in raw and raw["selection"] != selection:
                                raise ValueError("Saved selection differs from strict raw parsing")
                        saved = validate_target(raw["result"], list(FULL_FIELDS), frame)
                        if parsed != saved:
                            raise ValueError("Saved result differs from independently parsed raw output")
                        row.update(operational_ok=True, status=parsed["status"], result=parsed)
                        if parsed["status"] == "proposed":
                            row["point_in_original_mask"] = point_hit(parsed["destination_point_xy"], mask, frame["width"], frame["height"])
                except (ValueError, KeyError, TypeError) as exc:
                    row["error_type"] = "scoring_revalidation_error"
                    row["revalidation_error"] = str(exc)[:600]
            by_method[method].append(row)
        mask.close()

    coverable = {r["case_id"] for r in ceiling_rows if r["representative_point_covered"]}
    source_files = {"scorer": Path(__file__),
                    "selection_parser": Path(__file__).with_name("dropoff_method_trial.py"),
                    "direct_parser": Path(__file__).parents[1] / "trackrag/dropoff/formatting.py",
                    "geometry_contract": Path(__file__).parents[1] / "trackrag/dropoff/contracts.py",
                    "reference_loader": Path(__file__).parents[1] / "trackrag/dropoff/data.py"}
    result = {
        "schema_version": 1, "kind": "independent_saved_public_development_method_scoring",
        "case_ids": ids, "failures_unresolved_and_missing_retained_in_denominator": True,
        "public_masks_score_semantic_status_correctness": False,
        "candidate_targets_ever_sent_to_inference_by_scorer": False,
        "settings_declares_inference_source_hash": bool(settings.get("source_file_sha256")),
        "settings_source_hash_matches_current_trial_source": (settings["source_file_sha256"] == sha(source_files["selection_parser"].read_bytes())) if settings.get("source_file_sha256") else None,
        "source_bindings": {k: {"path": str(paths[k]), "bytes": len(v), "sha256": sha(v)} for k, v in snapshots.items()},
        "code_bindings": {k: {"path": str(p.resolve()), "sha256": sha(p.read_bytes())} for k, p in source_files.items()},
        "candidate_ceiling": {"requested": len(ids), "representative_point_covered": len(coverable),
                              "representative_point_coverage": len(coverable) / len(ids), "cases": ceiling_rows},
        "methods": {k: _summarize(rows, coverable) for k, rows in by_method.items()},
    }
    result["run_manifest"] = {"sha256": sha(snapshots["run_manifest"]), "complete": run.get("complete"),
                              "calls": run.get("calls"), "elapsed_s": run.get("elapsed_s")}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--outcomes", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("Choose a new immutable score output")
    result = score(args.cases, args.protocol, args.candidates, args.outcomes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"candidate_ceiling": result["candidate_ceiling"]["representative_point_covered"],
                      "requested": len(result["case_ids"]),
                      "methods": {k: {m: v[m] for m in ("point_hits", "proposals", "operational_errors")} for k, v in result["methods"].items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
