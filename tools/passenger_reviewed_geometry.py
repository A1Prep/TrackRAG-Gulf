"""Describe why native destination points missed the originally reviewed Gulf rectangles."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

from tools.passenger_reviewed_baseline import indexed, json_rows, reviewed_bank, score
from tools.dropoff_native_episodes import moment


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments/dropoff"


def area(box, width, height):
    left, top, right, bottom = box
    return (right - left) * (bottom - top) / (width * height)


def median_or_none(values):
    return median(values) if values else None


def position(value, lower, upper, before, after):
    if value < lower:
        return before
    if value > upper:
        return after
    return "within"


def describe(references, outcomes):
    horizontal = Counter()
    vertical = Counter()
    ref_areas, matched_areas, pred_areas = [], [], []
    errors_x, errors_y, positions = [], [], set()
    hits = 0
    for case_id, ref in references.items():
        if ref.target["status"] != "proposed":
            continue
        frame = ref.request["scene"]["frame"]
        width, height = frame["width"], frame["height"]
        box = ref.target["destination_box_xyxy"]
        fraction = area(box, width, height)
        ref_areas.append(fraction)
        outcome = outcomes[case_id]
        if not outcome["ok"] or outcome["result"]["status"] != "proposed":
            continue
        result = outcome["result"]
        x, y = result["destination_point_xy"]
        ref_x, ref_y = ref.target["destination_point_xy"]
        left, top, right, bottom = box
        x_position = position(x, left, right, "left", "right")
        y_position = position(y, top, bottom, "above", "below")
        horizontal[x_position] += 1
        vertical[y_position] += 1
        hits += x_position == "within" and y_position == "within"
        positions.add((round(x / width, 6), round(y / height, 6)))
        errors_x.append((x - ref_x) / width)
        errors_y.append((y - ref_y) / height)
        matched_areas.append(fraction)
        pred_areas.append(area(result["destination_box_xyxy"], width, height))
    return {
        "reviewed_location_cases": len(ref_areas),
        "model_returned_points": len(pred_areas),
        "missing_points": len(ref_areas) - len(pred_areas),
        "returned_points_within_both_axes": hits,
        "returned_points_horizontal": {k: horizontal[k] for k in ("left", "within", "right")},
        "returned_points_vertical": {k: vertical[k] for k in ("above", "within", "below")},
        "distinct_returned_normalized_points_six_decimals": len(positions),
        "median_signed_box_fraction_of_frame_all_reviewed": median_or_none(ref_areas),
        "median_signed_box_fraction_of_frame_on_returned_points": median_or_none(matched_areas),
        "median_predicted_box_fraction_of_frame_on_returned_points": median_or_none(pred_areas),
        "median_signed_horizontal_error_in_frame_widths_on_returned_points": median_or_none(errors_x),
        "median_signed_vertical_error_in_frame_heights_on_returned_points": median_or_none(errors_y),
    }


def run(requests, private_run, primary_path, eligibility):
    folders = {role: private_run / role for role in ("pretrained", "trained")}
    primary_bytes = primary_path.read_bytes()
    sealed = json.loads(primary_bytes)
    if score(requests, eligibility, private_run / "execution.json", folders) != sealed:
        raise ValueError("Native runs and reviewed inputs must reproduce the published primary scores")
    refs, _ = reviewed_bank(requests, moment(sealed["preanalysis"]["predeclared_utc"]))
    models = {}
    for role, folder in folders.items():
        raw = (folder / "model-outcomes.jsonl").read_bytes()
        primary = sealed["models"][role]
        if sha256(raw).hexdigest() != primary["native_file_sha256"]["model-outcomes.jsonl"]:
            raise ValueError("The native output changed after verification")
        metrics = describe(refs, indexed(json_rows(raw), "request_id"))
        if (metrics["returned_points_within_both_axes"], metrics["model_returned_points"]) != (
            primary["points_in_signed_box"], primary["returned_proposals_on_reference_proposals"]
        ):
            raise ValueError("Diagnostic and primary point counts differ")
        models[role] = metrics
    return {
        "kind": "posthoc_geometry_description_original_gulf_inputs",
        "primary_sha256": sha256(primary_bytes).hexdigest(),
        "scope": "Exploratory description after the primary results were known. No parameter fitting or new acceptance threshold.",
        "interpretation": "Signed rectangles are the original two-reviewer coarse targets. "
                          "Axis counts describe only cases with returned points. Both axes use inclusive boundaries. "
                          "Signed residuals subtract the reviewed point from the predicted point. "
                          "Positive values mean farther right or lower in the image.",
        "models": models,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--private-run", type=Path, required=True)
    parser.add_argument("--primary", type=Path, default=EXPERIMENTS / "passenger-reviewed-original-results-20261122.json")
    parser.add_argument("--eligibility", type=Path, default=EXPERIMENTS / "passenger-reviewed-original-eligibility-20261122.json")
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    data = (json.dumps(run(a.requests, a.private_run, a.primary, a.eligibility),
                       indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if a.out.exists():
        if a.out.read_bytes() != data:
            raise ValueError("Preserve the existing report and choose a fresh output path")
    else:
        with a.out.open("xb") as stream:
            stream.write(data)


if __name__ == "__main__":
    main()
