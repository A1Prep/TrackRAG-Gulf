"""Score frozen passenger outputs against genuine public navigable-region masks.

Usage: python -m tools.dropoff_passenger_references_public --corpus <unpacked bank>
       --outcomes <saved JSONL> --out <new JSON report>
"""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean

from PIL import Image
from tools.dropoff_passenger_references import _index, _lines, _payload
from tools.dropoff_predict import read_requests
from tools.dropoff_score_methods import point_hit
from trackrag.dropoff.contracts import DESTINATION_FIELDS, FULL_FIELDS, validate_target
from trackrag.dropoff.data import file_sha256, inference_input_sha256, load_cases
from trackrag.vision.geometry import iou


def score_outcome(case, request, outputs):
    row = {"request_id": case["case_id"], "source_scene": case["scene_group"], "failure": None,
           "status": None, "destination_provided": False, "point_in_public_mask": False,
           "coarse_box_iou": 0.0}
    if len(outputs) != 1:
        row["failure"] = "duplicate_output" if outputs else "missing_output"
        return row
    outcome = outputs[0]
    if outcome.get("ok") is False:
        error = outcome.get("error")
        row["failure"] = str(error.get("code") or "operational_error") if isinstance(error, dict) else "operational_error"
        return row
    try:
        if outcome.get("ok") is not True:
            raise ValueError("unrecognized grounding outcome")
        provenance = outcome["provenance"]
        digest = case["source"]["image_sha256"]
        if (provenance.get("injected") is True or provenance.get("independent_model_proposal") is False
                or provenance.get("method") == "explicit_passenger_region"):
            raise ValueError("artificial or passenger-marked outputs cannot score as an independent model")
        if provenance["image"]["sha256"] != digest:
            raise ValueError("output belongs to different image bytes")
        if "inference_input_sha256" in provenance:
            if provenance["inference_input_sha256"] != inference_input_sha256(request, digest):
                raise ValueError("wrong portable input digest")
        else:
            model_input = json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            if provenance.get("request_sha256") != sha256(model_input).hexdigest():
                raise ValueError("wrong request digest")
        frame = request["scene"]["frame"]
        result = outcome["result"]
        if set(result) != DESTINATION_FIELDS and not FULL_FIELDS.issubset(result):
            raise ValueError("unrecognized model result fields")
        result = validate_target(result, list(result), frame)
        # "destination_only" describes the output contract, not a reference label.
        row["status"] = result.get("status", "destination_only")
        if result["destination_point_xy"] is not None:
            with Image.open(case["source"]["mask_path"]) as reference:
                hit = point_hit(result["destination_point_xy"], reference.convert("L"), frame["width"], frame["height"])
            row.update(destination_provided=True, point_in_public_mask=bool(hit),
                       coarse_box_iou=iou(result["destination_box_xyxy"], case["target"]["destination_box_xyxy"]),
                       model_point=result["destination_point_xy"], model_box=result["destination_box_xyxy"])
    except (ValueError, OSError, TypeError, KeyError) as exc:
        row.update(status=None, failure="invalid_or_unbound_output", detail=str(exc))
    return row


def score(corpus, outcomes):
    meta = json.loads(_payload(corpus / "manifest.json"))
    reference_file = corpus / "corpus" / "references.jsonl"
    input_file = corpus / "inputs" / "requests.jsonl"
    if (file_sha256(reference_file) != meta["reference_file_sha256"]
            or file_sha256(input_file) != meta["request_file_sha256"]
            or file_sha256(corpus / "selection-protocol.json") != meta["protocol_sha256"]):
        raise ValueError("content differs from the frozen public source diagnostic")
    requests = _index(read_requests(input_file))
    cases = load_cases(reference_file, allowed_partitions={"development"})
    if len(cases) != meta["cases"] or {c["case_id"] for c in cases} != set(requests):
        raise ValueError("public cohort identities differ")
    if (len(cases) != len({c["scene_group"] for c in cases})
            or [c["source"]["source_index"] for c in cases] != meta["source_indices"]
            or [c["source"]["scene_token"] for c in cases] != meta["source_scenes"]):
        raise ValueError("frozen source-scene membership differs")
    saved = _payload(outcomes)
    outputs = defaultdict(list)
    for record in _lines(saved):
        if not isinstance(record, dict) or not isinstance(record.get("request_id"), str):
            raise ValueError("saved output cannot be attributed")
        outputs[record["request_id"]].append(record)
    rows = []
    for case in cases:
        request = requests[case["case_id"]]
        frame = request["scene"]["frame"]
        digest = case["source"]["image_sha256"]
        if (case["source"]["dataset"] != "Talk2Car-RegSeg" or case["review"]["status"] != "public_annotation"
                or set(case["target"]) != DESTINATION_FIELDS or request["scene"]["source"] != "predicted"
                or request["scene"]["tracks"] or sha256(Path(frame["image_path"]).read_bytes()).hexdigest() != digest
                or request["scene"]["provenance"].get("image_sha256") != digest
                or inference_input_sha256(request, digest) != inference_input_sha256(case["request"], digest)):
            raise ValueError("image, command, reference labels or zero-track condition differ")
        rows.append(score_outcome(case, request, outputs[case["case_id"]]))
    return {"kind": "new_scene_public_destination_geometry", "cases": len(rows),
            "source_groups": len({r["source_scene"] for r in rows}), "semantic_status_ground_truth": False,
            "source_reference": "Original authors' manually annotated Talk2Car-RegSeg navigable-region masks",
            "source_paper": "https://arxiv.org/abs/2112.13031", "selection_protocol_sha256": meta["protocol_sha256"],
            "input_file_sha256": meta["request_file_sha256"], "references_sha256": meta["reference_file_sha256"],
            "outcomes_sha256": sha256(saved).hexdigest(),
            "point_hits_all_requests": sum(r["point_in_public_mask"] for r in rows),
            "destinations_provided": sum(r["destination_provided"] for r in rows),
            "mean_coarse_box_iou_all_requests_missing_as_zero": mean(r["coarse_box_iou"] for r in rows),
            "output_status_counts": dict(Counter(r["status"] for r in rows if r["status"])),
            "failures": dict(Counter(r["failure"] for r in rows if r["failure"])),
            "unexpected_output_ids": sorted(outputs.keys() - requests.keys()), "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--outcomes", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    summary = score(args.corpus, args.outcomes)
    with args.out.open("x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
