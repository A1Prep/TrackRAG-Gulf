"""Post-score geometry provenance check. Emits cohort aggregates only, with no inference or metric changes."""

import argparse
from collections import Counter
from fractions import Fraction
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from statistics import median, quantiles
import subprocess
import tarfile
from tempfile import TemporaryDirectory

from evaluation.grounding import box_iou
from trackrag.emt import load_annotated_tracks, open_recording


ROOT = Path(__file__).resolve().parents[1]
TASK2 = "2d96dcb6afdffb46c5d94d299768fe6a92acc094"
DRAFT = "186a80eebf02e797ca5161ed42a77f24c6eada7d"
PACKAGE = "Task2_56_Case_Human_Review_Corrected"
EXECUTION = "a59e25f525dceb57e4e8ec82e62ebceb29f98ccf229aa54885ef3738d09a6a38"
SIGNED = "37f21c01f44f1a35fffc8092285d997713965dbad22086ac3a3de2fcb0557373"
ANNOTATIONS = "a5d6bb56995738acd1251ffcdfea11432a2f47d96e1da5e3c34897debde45372"


def require(value, message):
    if not value:
        raise ValueError(message)


def content_sha(data):
    return sha256(data).hexdigest()


def rows(data):
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def inside(point, box):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def shape_stats(proposals):
    sizes, boxes, ratios, points, areas = [], [], [], [], []
    centered = contained = aspect_square = 0
    for case in proposals:
        frame, target = case["request"]["scene"]["frame"], case["target"]
        w, h = frame["width"], frame["height"]
        x, y = map(lambda n: Fraction(str(n)), target["destination_point_xy"])
        left, top, right, bottom = map(lambda n: Fraction(str(n)), target["destination_box_xyxy"])
        points.append((x / w, y / h))
        boxes.append((left / w, top / h, right / w, bottom / h))
        sizes.append(((right - left) / w, (bottom - top) / h))
        ratios.append((right - left) / (bottom - top))
        areas.append(sizes[-1][0] * sizes[-1][1])
        centered += x * 2 == left + right and y * 2 == top + bottom
        contained += inside((x, y), (left, top, right, bottom))
        aspect_square += right - left == bottom - top

    def counts(values):
        seen = Counter(values)
        return {
            "unique": len(seen),
            "most_frequent_count": max(seen.values()),
            "cases_sharing_a_value_with_another_case": sum(c for c in seen.values() if c > 1),
        }

    def distribution(values):
        q1, center, q3 = quantiles(values, n=4, method="inclusive")
        return {
            "q1": round(float(q1), 6),
            "median": round(float(center), 6),
            "q3": round(float(q3), 6),
        }

    return {
        "eligible": len(proposals),
        "signed_point_inside_own_box": contained,
        "signed_point_exactly_at_box_midpoint": centered,
        "square_boxes_in_original_pixels": aspect_square,
        "normalized_box_sizes": counts(sizes),
        "normalized_complete_boxes": counts(boxes),
        "normalized_points": counts(points),
        "box_to_frame_area_fraction": distribution(areas),
        "width_fraction": distribution([s[0] for s in sizes]),
        "height_fraction": distribution([s[1] for s in sizes]),
        "point_x_fraction": distribution([v[0] for v in points]),
        "point_y_fraction": distribution([v[1] for v in points]),
        "original_box_width_height_ratio": distribution(ratios),
    }


def original_object_boxes(keys, archive_bytes, image_root):
    """Read original frame and track IDs with the same annotation reader as Task 2."""
    objects = {key: {} for key in keys}
    with (
        tarfile.open(fileobj=BytesIO(archive_bytes), mode="r:gz") as packed,
        TemporaryDirectory() as directory,
    ):
        extracted = Path(directory) / "source.txt"
        for recording_id in sorted({key[0] for key in keys}):
            recording = open_recording(image_root, recording_id, split="test")
            with packed.extractfile(f"EMT/annotations/test/{recording_id}.txt") as member:
                extracted.write_bytes(member.read())
            for observation in load_annotated_tracks(recording, extracted):
                key = (recording_id, observation["frame_id"])
                if key in objects:
                    objects[key][observation["track_id"]] = observation["bbox_xyxy"]
    return objects


def object_comparison(proposals, objects):
    totals = Counter()
    max_ious = []
    for case in proposals:
        key = (case["source"]["recording_id"], case["source"]["frame_id"])
        labels = objects[key]
        listed = {landmark["track_id"] for landmark in case["source"]["landmarks"]}
        target = case["target"]
        box = tuple(target["destination_box_xyxy"])
        point = target["destination_point_xy"]
        available = [labels[i] for i in listed if i in labels]
        totals["frames_with_listed_landmarks"] += bool(listed)
        totals["listed_ids"] += len(listed)
        totals["listed_ids_found_in_official_source"] += len(available)
        totals["frames_with_at_least_one_matching_listed_id"] += bool(available)
        totals["frames_with_all_listed_ids_matched"] += bool(listed) and len(available) == len(
            listed
        )
        totals["signed_rectangle_exactly_equal_to_any_official_road_user_rectangle"] += (
            box in labels.values()
        )
        totals["signed_rectangle_exactly_equal_to_a_listed_landmark_rectangle"] += box in available
        totals["signed_point_inside_any_official_road_user_bounding_rectangle"] += any(
            inside(point, b) for b in labels.values()
        )
        totals["signed_point_inside_a_matching_listed_landmark_rectangle"] += any(
            inside(point, b) for b in available
        )
        if available:
            best = max(box_iou(box, landmark_box) for landmark_box in available)
            max_ious.append(best)
            totals["eligible_frames_with_any_positive_box_overlap_with_listed_landmark"] += best > 0
            totals["eligible_frames_with_a_listed_landmark_box_iou_at_least_half"] += best >= 0.5
    return {
        **totals,
        "median_max_iou_with_matched_listed_landmark": round(median(max_ious), 6)
        if max_ious
        else None,
        "comparison": "Original official scene annotations matched by recording, frame and source track ID. Object rectangles overlap in projection and do not measure road-surface pixels.",
    }


def native_comparison(folder, signed, expected_hash):
    manifest_bytes = (folder / "run.json").read_bytes()
    require(content_sha(manifest_bytes) == expected_hash, "Native manifest differs")
    manifest = json.loads(manifest_bytes)
    out = (folder / "model-outcomes.jsonl").read_bytes()
    rec = (folder / "receipts.jsonl").read_bytes()
    require(
        content_sha(out) == manifest["outcomes_sha256"]
        and content_sha(rec) == manifest["receipts_sha256"],
        "Native outputs differ",
    )
    result_rows, receipt_rows = rows(out), rows(rec)
    require(len(result_rows) == len(receipt_rows) == len(signed) == 44, "Incomplete native cohort")
    return compare_geometry(signed, result_rows, receipt_rows)


def compare_geometry(signed, outcomes, receipts):
    """Compare the original normalized model answer independently with original-frame pixel geometry."""
    total = Counter()
    matched_points = []
    paired_errors = []
    misses = Counter()
    observed_ids = set()
    max_error = 0
    require(len(outcomes) == len(receipts) == len(signed), "Incomplete comparison")
    for outcome, receipt in zip(outcomes, receipts, strict=True):
        ident = outcome["request_id"]
        case = signed[ident]
        f = case["request"]["scene"]["frame"]
        gold = case["target"]
        w, h = f["width"], f["height"]
        require(
            receipt["request_id"] == ident and ident not in observed_ids,
            "Missing, mismatched or duplicate receipt",
        )
        observed_ids.add(ident)
        if not outcome["ok"] or outcome["result"]["status"] != "proposed":
            continue
        raw = receipt["raw_text"].strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.splitlines()[1:-1])
        original = json.loads(raw)
        converted = outcome["result"]
        total["native_proposals_all_44"] += 1
        for field in ("destination_point_xy", "destination_box_xyxy"):
            expected = [n * (w if i % 2 == 0 else h) / 1000 for i, n in enumerate(original[field])]
            require(len(expected) == len(converted[field]), "Unexpected geometry shape")
            max_error = max(
                max_error,
                max(
                    abs(left - right)
                    for left, right in zip(expected, converted[field], strict=True)
                ),
            )
        total["returned_point_inside_own_returned_box"] += inside(
            converted["destination_point_xy"], converted["destination_box_xyxy"]
        )
        if gold["status"] == "proposed":
            total["native_proposals_on_33_eligible"] += 1
            point = converted["destination_point_xy"]
            box = gold["destination_box_xyxy"]
            hits = inside(point, box)
            total["hits_in_original_pixels"] += hits
            if not hits:
                horizontal = not box[0] <= point[0] <= box[2]
                vertical = not box[1] <= point[1] <= box[3]
                axis = (
                    "both"
                    if horizontal and vertical
                    else ("horizontal_only" if horizontal else "vertical_only")
                )
                misses[axis] += 1
            normalized_box = [
                v * 1000 / (w if i % 2 == 0 else h)
                for i, v in enumerate(gold["destination_box_xyxy"])
            ]
            total["hits_independently_in_normalized_raw_model_space"] += inside(
                original["destination_point_xy"], normalized_box
            )
            require(
                hits == inside(original["destination_point_xy"], normalized_box),
                "Pixel and normalized containment disagree",
            )
            matched_points.append((point[0] / w, point[1] / h))
            gold_x, gold_y = gold["destination_point_xy"]
            paired_errors.append(((point[0] - gold_x) / w, (point[1] - gold_y) / h))
    require(observed_ids == set(signed), "One or more source requests were not compared")
    return {
        **total,
        "largest_reconstructed_coordinate_difference_original_pixels": max_error,
        "median_normalized_returned_point_x_on_eligible_proposals": round(
            median(x for x, _ in matched_points), 6
        )
        if matched_points
        else None,
        "median_normalized_returned_point_y_on_eligible_proposals": round(
            median(y for _, y in matched_points), 6
        )
        if matched_points
        else None,
        "miss_axis_among_returned_points_on_eligible_proposals": dict(misses),
        "median_absolute_horizontal_error_as_image_width_fraction_among_returned": round(
            median(abs(x) for x, _ in paired_errors), 6
        )
        if paired_errors
        else None,
        "median_absolute_vertical_error_as_image_height_fraction_among_returned": round(
            median(abs(y) for _, y in paired_errors), 6
        )
        if paired_errors
        else None,
    }


def earlier_draft_comparison(proposed):
    source_path = "task3 handoffs files/batch1_eval_for_task3.jsonl"
    raw = subprocess.check_output(["git", "show", f"{DRAFT}:{source_path}"], cwd=ROOT)
    earlier = rows(raw)
    current = {
        (case["source"]["recording_id"], case["source"]["frame_id"]): case for case in proposed
    }
    totals = Counter()
    for draft in earlier:
        case = current.get((draft["recording_id"], draft["anchor_frame_id"]))
        if case is None:
            continue
        totals["same_recording_and_frame"] += 1
        frame = case["request"]["scene"]["frame"]
        require(
            draft["image_size_hw"] == [frame["height"], frame["width"]],
            "Overlapping draft has a different pixel basis",
        )
        totals["also_same_instruction"] += draft["instruction"] == case["request"]["text"]
        totals["same_point"] += (
            draft["destination"]["point"] == case["target"]["destination_point_xy"]
        )
        totals["same_rectangle"] += (
            draft["destination"]["box_xyxy"] == case["target"]["destination_box_xyxy"]
        )
    return {"draft_cases": len(earlier), "sha256": content_sha(raw), **totals}


def audit(gulf, annotations, image_root):
    source = subprocess.check_output(
        ["git", "show", f"{TASK2}:{PACKAGE}/review_cases.jsonl"], cwd=ROOT
    )
    execution_bytes = (gulf / "execution.json").read_bytes()
    require(
        content_sha(source) == SIGNED and content_sha(execution_bytes) == EXECUTION,
        "Expected original signed set and paired execution",
    )
    signed = {case["case_id"]: case for case in rows(source) if case["partition"] == "test"}
    proposed = [case for case in signed.values() if case["target"]["status"] == "proposed"]
    require(len(signed) == 44 and len(proposed) == 33, "Unexpected cohort")
    annotation_bytes = annotations.read_bytes()
    require(
        content_sha(annotation_bytes) == ANNOTATIONS, "Original official source archive differs"
    )
    keys = {(case["source"]["recording_id"], case["source"]["frame_id"]) for case in proposed}
    objects = original_object_boxes(keys, annotation_bytes, image_root)
    state = json.loads(execution_bytes)
    github = "https://github.com/A1Prep/TrackRAG-Gulf/blob/"
    package_url = f"{github}{TASK2}/{PACKAGE}"
    return {
        "kind": "post_score_aggregate_geometry_and_provenance_diagnostic",
        "comparison_changes": False,
        "individual_rows_exported": False,
        "model_calls": 0,
        "task2_commit": TASK2,
        "source_case_sha256": SIGNED,
        "native_execution_sha256": EXECUTION,
        "emt_test_annotation_sha256": ANNOTATIONS,
        "declared_annotation_basis_counts": dict(
            Counter(q["source"]["annotation_basis"] for q in signed.values())
        ),
        "original_image_sizes_on_proposed_cases": dict(
            Counter(
                f"{q['request']['scene']['frame']['width']}x{q['request']['scene']['frame']['height']}"
                for q in proposed
            )
        ),
        "signed_geometry": shape_stats(proposed),
        "source_object_boxes": object_comparison(proposed, objects),
        "earlier_local_draft_geometry_on_overlapping_frames": earlier_draft_comparison(proposed),
        "primary_source_links": {
            "corrected_review_task_and_original_pixel_convention": f"{package_url}/REVIEW_INSTRUCTIONS.md#L16-L29",
            "separation_of_landmarks_and_ground_geometry": f"{package_url}/REPOSITORY_ALIGNMENT.md#L5-L7",
            "status_of_rechecked_and_changed_candidates": f"{package_url}/CORRECTION_REPORT.md#L5-L15",
            "ai_assisted_corrections_before_original_confirmation": f"{package_url}/manifest.json#L2",
            "resolver_keeps_existing_candidate_on_two_accepts": f"{package_url}/tools/dropoff_resolve_reviews.py#L233-L251",
            "older_authored_circle_description": f"{github}{DRAFT}/task5%20handoffs-task2/field_provenance.md.md#L22-L30",
            "older_local_release_target_fields": f"{github}{DRAFT}/trackrag/dropoff.py#L368-L378",
            "original_emt_annotations": "https://huggingface.co/datasets/KuAvLab/EMT/blob/4418cec01b03c6f797cadb06c9405843a30988e2/test_annotation.tar.gz",
            "runtime_normalized_to_original_pixel_conversion": f"{github}d99f2fc35dcb61c2e30686220f4dc047a8a9f595/trackrag/dropoff/formatting.py#L128-L157",
            "original_image_pixel_hit_rule": f"{github}d99f2fc35dcb61c2e30686220f4dc047a8a9f595/tools/passenger_reviewed_baseline.py#L315-L328",
        },
        "native_geometry": {
            role: native_comparison(gulf / role, signed, v["run_sha256"])
            for role, v in state["conditions"].items()
        },
        "notes": [
            "Signed gold means independent acceptance of the bound coarse point/rectangle. The corrected package does not contain per-reviewer raw drawing gestures or a stopping-area polygon.",
            "An earlier handoff describes authored circle masks. The earlier general export kept local track IDs and landmarks. A separate 11-case draft includes geometry, but the points and rectangles on its seven overlapping frames all differ from the corrected candidates. The current handoff does not establish how each rectangle was first drawn or computed.",
            "This diagnostic checks xyxy, dimensions and the 0–1000 conversion for the actual raw proposals. A model can still misunderstand coordinates or passenger intent, and an accepted coarse rectangle is not a complete map of all plausible stopping places.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gulf", type=Path, default=ROOT / "runs/task5-independent-typed-20261122/gulf"
    )
    parser.add_argument(
        "--official-labels",
        type=Path,
        default=ROOT / "data/task1-annotation-source-20261119/test_annotation.tar.gz",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--image-root",
        type=Path,
        default=ROOT / "runs/task5-evaluator-delivery-20261122/installed/inputs/source",
    )
    args = parser.parse_args()
    report = audit(args.gulf, args.official_labels, args.image_root)
    payload = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    with args.out.open("xb") as f:
        f.write(payload)
