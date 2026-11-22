"""Score saved typed passenger outcomes against the frozen AI-authored diagnostic banks.

This offline comparison reads no checkpoints, calls no models and never changes references
or predictions. Each reference rectangle describes one author's interpretation of a visible
destination. A point inside it is agreement with that rectangle, not a safety determination.
"""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean, median

from tools.dropoff_predict import read_requests
from trackrag.dropoff.contracts import FULL_FIELDS, validate_target
from trackrag.dropoff.data import file_sha256, inference_input_sha256
from trackrag.vision.geometry import iou


CONTRASTS = {
    "train": (("a01", "a02"), ("a06", "a07"), ("a10", "a11"), ("a15", "a16")),
    "development": (("b01", "b02"), ("b06", "b07"), ("b11", "b12"), ("b16", "b17")),
    "night": (("c01", "c02"), ("c04", "c05"), ("c09", "c10"), ("c13", "c14")),
}
POINT_ERROR_TOLERANCES = (0.025, 0.05)


def _unique(pairs):
    row = {}
    for key, value in pairs:
        if key in row:
            raise ValueError(f"duplicate JSON field {key}")
        row[key] = value
    return row


def _nonfinite(_):
    raise ValueError("nonfinite JSON input")


def _json(raw):
    return json.loads(raw, object_pairs_hook=_unique, parse_constant=_nonfinite)


def _payload(path):
    if path.stat().st_size > 64 * 1024**2:
        raise ValueError(f"{path.name} exceeds the diagnostic file limit")
    return path.read_bytes()


def _lines(raw):
    return [_json(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]


def _index(rows):
    result = {}
    for row in rows:
        identity = row.get("request_id") if isinstance(row, dict) else None
        if not isinstance(identity, str) or not identity.strip() or identity in result:
            raise ValueError("duplicate or missing request identity in the frozen inputs")
        result[identity] = row
    return result


def _frozen(bank, freeze, name):
    entry = freeze["files"][name]
    expected = entry["sha256"] if isinstance(entry, dict) else entry
    data = _payload(bank / name)
    normalized = data.replace(b"\r\n", b"\n")
    if sha256(normalized).hexdigest() != expected:
        raise ValueError(f"frozen content differs for {name}")
    return _lines(normalized)


def gold_and_requests(bank, partition, requests_path):
    freeze_bytes = _payload(bank / "freeze.json")
    freeze = _json(freeze_bytes)
    receipt = _json(_payload(bank / "input-bindings-receipt.json"))
    raw_bindings = _payload(bank / "input-bindings.jsonl").replace(b"\r\n", b"\n")
    if (sha256(raw_bindings).hexdigest() != receipt["bindings_file_sha256"]
            or sha256(freeze_bytes).hexdigest() != receipt["freeze_file_sha256"]):
        raise ValueError("portable input bindings disagree with the original dataset freeze")
    bindings = _index(_lines(raw_bindings))
    prefix = "" if partition == "night" else partition + "-"
    specs = _index(_frozen(bank, freeze, prefix + "instructions.jsonl"))
    refs = _index(_frozen(bank, freeze, prefix + "references.jsonl"))
    frames = {(r["recording_id"], r["frame_id"]): r
              for r in _frozen(bank, freeze, "frames.jsonl")}
    payload = _payload(requests_path)
    requests = _index(read_requests(requests_path, payload=payload))
    if not specs or not refs.keys() == specs.keys() == requests.keys():
        raise ValueError("requested, frozen specification and reference case IDs must match exactly")
    verified_images = {}
    for identity, ref in refs.items():
        spec, request = specs[identity], requests[identity]
        frame = request["scene"]["frame"]
        origin = (ref["recording_id"], ref["frame_id"])
        if (origin != (frame["recording_id"], frame["frame_id"])
                or origin != (spec["recording_id"], spec["frame_id"])):
            raise ValueError(f"incorrect scene identity for {identity}")
        if (request["input_mode"] != "typed" or request["text"] != spec["text"]
                or request["dialogue"] != spec.get("dialogue", [])):
            raise ValueError(f"wrong typed instruction or prior dialogue for {identity}")
        src = frames[origin]
        if ((frame["width"], frame["height"]) != (src["width"], src["height"])
                or any(track["frame_id"] != frame["frame_id"] for track in request["scene"]["tracks"])):
            raise ValueError(f"incorrect dimensions or noncurrent track for {identity}")
        image = Path(frame["image_path"])
        if image not in verified_images:
            verified_images[image] = file_sha256(image)
        digest = src["image_sha256"]
        provenance = request["scene"].get("provenance", {})
        if (verified_images[image] != digest or ref["image_sha256"] != digest
                or provenance.get("image_sha256") != digest
                or provenance.get("tracks_sha256") != src["saved_predictions_sha256"]
                or provenance.get("prediction_provenance_sha256") != src["saved_prediction_provenance_sha256"]
                or provenance.get("vision_identity_hash") != src["vision_identity_hash"]):
            raise ValueError(f"frozen image or saved prediction provenance differs for {identity}")
        if len(request["scene"]["tracks"]) != src["current_frame_predicted_track_count"]:
            raise ValueError(f"saved current-track count differs for {identity}")
        if inference_input_sha256(request, digest) != bindings[identity]["inference_input_sha256"]:
            raise ValueError(f"actual input differs from the independently verified saved current-frame rows for {identity}")
        validate_target(ref["target"], sorted(FULL_FIELDS), frame)
    return refs, requests, sha256(payload).hexdigest()


def _row(ref, request, outputs):
    key = ref["request_id"]
    target = ref["target"]
    row = {"request_id": key, "recording_id": ref["recording_id"], "frame_id": ref["frame_id"],
           "expected_status": target["status"], "status": None, "failure": None,
           "status_matches": False, "point_in_authored_box": False, "normalized_point_error": None,
           "box_iou": None, "clarification_question": None}
    if not outputs:
        row["failure"] = "missing_output"
        return row
    if len(outputs) != 1:
        row["failure"] = "duplicate_output"
        return row
    outcome = outputs[0]
    if outcome.get("ok") is False:
        error = outcome.get("error")
        row["failure"] = (str(error.get("code") or "operational_error")
                          if isinstance(error, dict) else "operational_error")
        return row
    try:
        if outcome.get("ok") is not True:
            raise ValueError("unrecognized outcome type")
        provenance = outcome["provenance"]
        if provenance.get("injected") is True:
            raise ValueError("Injected software-check output cannot score as an independent model")
        if provenance.get("independent_model_proposal") is False or provenance.get("method") == "explicit_passenger_region":
            raise ValueError("passenger marked or intervention-assisted output cannot score as model-only")
        digest = ref["image_sha256"]
        if provenance["image"]["sha256"] != digest:
            raise ValueError("wrong saved image")
        portable = inference_input_sha256(request, digest)
        if "inference_input_sha256" in provenance:
            if provenance["inference_input_sha256"] != portable:
                raise ValueError("wrong portable input binding")
        else:
            # The initial passenger prototype recorded path-bound canonical request hashes only.
            exact = sha256(json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            if provenance.get("request_sha256") != exact:
                raise ValueError("wrong instruction/scene binding")
        result = outcome["result"]
        if not FULL_FIELDS.issubset(result):
            raise ValueError("partial model result cannot count as complete interpretation")
        result = validate_target(result, list(result), request["scene"]["frame"])
        row.update(status=result["status"], status_matches=result["status"] == target["status"],
                   clarification_question=result["clarification_question"])
        if target["status"] == "proposed" and result["status"] == "proposed":
            x, y = result["destination_point_xy"]
            l, t, r, b = target["destination_box_xyxy"]
            frame = request["scene"]["frame"]
            row.update(point_in_authored_box=l <= x < r and t <= y < b,
                       predicted_point=[x, y], predicted_box=result["destination_box_xyxy"],
                       normalized_point_error=math.dist([x, y], target["destination_point_xy"]) /
                                              math.hypot(frame["width"], frame["height"]),
                       box_iou=iou(result["destination_box_xyxy"], target["destination_box_xyxy"]))
    except (KeyError, TypeError, ValueError) as exc:
        row.update(status=None, status_matches=False, failure="invalid_or_mismatched_output",
                   detail=str(exc))
    return row


def score(bank, partition, requests_path, outputs_path):
    refs, inputs, request_hash = gold_and_requests(bank, partition, requests_path)
    raw = _payload(outputs_path)
    outputs = defaultdict(list)
    for value in _lines(raw):
        if not isinstance(value, dict) or not isinstance(value.get("request_id"), str):
            raise ValueError("saved output cannot be attributed to a request")
        outputs[value["request_id"]].append(value)
    extras = sorted(outputs.keys() - refs.keys())
    rows = [_row(ref, inputs[key], outputs[key]) for key, ref in refs.items()]
    proposed = [r for r in rows if r["expected_status"] == "proposed"]
    joint = [r for r in proposed if r["status"] == "proposed"]
    expected = Counter(r["expected_status"] for r in rows)
    confusion = {k: dict(Counter(r["status"] or "failure" for r in rows if r["expected_status"] == k))
                 for k in sorted(expected)}
    by_id = {r["request_id"]: r for r in rows}
    contrasts = []
    for a, b in CONTRASTS[partition]:
        keys = ["passenger-" + a, "passenger-" + b]
        first, second = (by_id[k] for k in keys)
        if ((first["recording_id"], first["frame_id"]) != (second["recording_id"], second["frame_id"])
                or any(row["expected_status"] != "proposed" for row in (first, second))):
            raise ValueError("predeclared contrast must be two distinct proposed meanings on one frame")
        both = first["status"] == second["status"] == "proposed"
        closer = False
        if both:
            x, y = first["predicted_point"], second["predicted_point"]
            p, q = (refs[k]["target"]["destination_point_xy"] for k in keys)
            closer = math.dist(x, p) < math.dist(x, q) and math.dist(y, q) < math.dist(y, p)
        contrasts.append({"requests": keys, "both_proposed": both, "each_closer_to_own_provisional_point": closer,
                          "both_in_authored_rectangle": first["point_in_authored_box"] and second["point_in_authored_box"]})
    wrong_status_proposals = sum(r["status"] == "proposed" and r["expected_status"] != "proposed" for r in rows)
    point_misses = sum(not r["point_in_authored_box"] for r in joint)
    return {"kind": "offline_model_only_versus_ai_authored_rectangles", "partition": partition,
            "input_file_sha256": request_hash, "outputs_sha256": sha256(raw).hexdigest(),
            "freeze_file_sha256": sha256(_payload(bank / "freeze.json")).hexdigest(),
            "requested": len(rows), "expected_status_counts": dict(expected),
            "status_matches": sum(r["status_matches"] for r in rows), "confusion": confusion,
            "failures": dict(Counter(r["failure"] for r in rows if r["failure"])),
            "unexpected_output_ids": extras, "proposal_count": sum(r["status"] == "proposed" for r in rows),
            "proposals_on_nonproposal_cases": wrong_status_proposals,
            "clear_reference_count": len(proposed), "clear_reference_and_model_proposal_count": len(joint),
            "point_in_authored_box_count": sum(r["point_in_authored_box"] for r in proposed),
            "missed_authored_rectangle_on_paired_proposals": point_misses,
            "proposal_on_wrong_status_or_outside_authored_rectangle": wrong_status_proposals + point_misses,
            "normalized_tolerance_hits_all_clear": {
                str(tolerance): sum(r["normalized_point_error"] <= tolerance for r in joint)
                for tolerance in POINT_ERROR_TOLERANCES},
            "mean_box_iou_all_clear_missing_as_zero": mean(r["box_iou"] or 0 for r in proposed) if proposed else None,
            "median_normalized_point_error_paired_proposals_only": median(r["normalized_point_error"] for r in joint) if joint else None,
            "same_frame_contrasts": contrasts,
            "per_case": rows, "human_reference_reviews": 0, "final_evaluation": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--partition", required=True, choices=["train", "development", "night"])
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = score(args.bank, args.partition, args.requests, args.outcomes)
    with args.out.open("x", encoding="utf-8", newline="\n") as dest:
        dest.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "per_case"}, indent=2))


if __name__ == "__main__":
    main()
