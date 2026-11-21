#!/usr/bin/env python3
"""Merge current-binding human reviews and apply only mutual approvals."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path

ALLOWED = {"accept", "reject", "correct", "unavailable"}
STATUSES = {"proposed", "needs_clarification", "insufficient_visual_evidence"}
HEADER = [
    "case_id", "partition", "recording_id", "frame_id", "image_available", "image_path",
    "image_sha256", "instruction", "status", "destination_point_xy", "destination_box_xyxy",
    "clarification_question", "input_sha256", "supervision_sha256", "decision",
    "corrected_status", "corrected_destination_point_xy", "corrected_destination_box_xyxy",
    "corrected_clarification_question", "reviewer_id", "reviewed_at", "notes",
    "legacy_decision", "legacy_corrected_destination_ref", "legacy_notes",
]
CORRECTIONS = [
    "corrected_status", "corrected_destination_point_xy", "corrected_destination_box_xyxy",
    "corrected_clarification_question",
]
RESOLUTION_HEADER = [
    "case_id", "reviewer_1_decision", "reviewer_2_decision", "reviewer_1_id", "reviewer_2_id",
    "reviewer_1_reviewed_at", "reviewer_2_reviewed_at", "reviewer_1_correction",
    "reviewer_2_correction", "resolution", "resolution_notes",
]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def normalized_target(target, fields, frame):
    if set(fields) != {"status", "destination_point_xy", "destination_box_xyxy", "clarification_question"}:
        raise ValueError("Only complete local interpretation supervision is supported")
    if not isinstance(target, dict) or set(target) != set(fields) or target.get("status") not in STATUSES:
        raise ValueError("Invalid target fields or status")
    result = deepcopy(target)
    width, height = frame["width"], frame["height"]
    if target["status"] == "proposed":
        point, box = target["destination_point_xy"], target["destination_box_xyxy"]
        if not isinstance(point, list) or len(point) != 2 or not isinstance(box, list) or len(box) != 4:
            raise ValueError("A proposal requires a point and box")
        try:
            point = [float(v) for v in point]
            box = [float(v) for v in box]
        except (TypeError, ValueError) as error:
            raise ValueError("Coordinates must be finite numbers") from error
        if not all(v == v and abs(v) != float("inf") for v in point + box):
            raise ValueError("Coordinates must be finite numbers")
        left, top, right, bottom = box
        if not (0 <= point[0] < width and 0 <= point[1] < height):
            raise ValueError("Destination point is outside the image")
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError("Destination box is empty or outside the image")
        if not (left <= point[0] <= right and top <= point[1] <= bottom):
            raise ValueError("Destination point is outside its box")
        if target["clarification_question"] is not None:
            raise ValueError("A proposal cannot contain a clarification question")
        result["destination_point_xy"], result["destination_box_xyxy"] = point, box
    else:
        if target["destination_point_xy"] is not None or target["destination_box_xyxy"] is not None:
            raise ValueError("An unresolved outcome cannot contain geometry")
        question = target["clarification_question"]
        if target["status"] == "needs_clarification":
            if not isinstance(question, str) or not question.strip():
                raise ValueError("Clarification requires a useful question")
        elif question is not None:
            raise ValueError("Insufficient evidence cannot contain a question")
    return result


def supervision_sha256(case):
    value = normalized_target(case["target"], case["supervised_fields"], case["request"]["scene"]["frame"])
    payload = {"target": value, "supervised_fields": sorted(case["supervised_fields"])}
    return hashlib.sha256(canonical(payload)).hexdigest()


def input_sha256(case):
    value = deepcopy(case["request"])
    frame = value["scene"]["frame"]
    frame.pop("image_path")
    frame["image_sha256"] = case["source"]["image_sha256"].lower()
    value["scene"].pop("provenance", None)
    return hashlib.sha256(canonical(value)).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def read_form(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != HEADER:
            raise ValueError(f"{path}: review header does not match the current schema")
        rows = list(reader)
    result = {}
    for row in rows:
        case_id = row["case_id"]
        if case_id in result:
            raise ValueError(f"Duplicate review for {case_id}")
        decision = row["decision"].strip().lower()
        if decision and decision not in ALLOWED:
            raise ValueError(f"{case_id}: decision must be blank, accept, reject, correct, or unavailable")
        row["decision"] = decision
        if not decision:
            if any(row[field].strip() for field in CORRECTIONS + ["reviewer_id", "reviewed_at"]):
                raise ValueError(f"{case_id}: a pending row cannot contain a reviewer, timestamp, or correction")
            result[case_id] = row
            continue
        if decision == "unavailable":
            if row["image_available"].lower() != "false":
                raise ValueError(f"{case_id}: unavailable is reserved for missing images")
        else:
            if row["image_available"].lower() != "true":
                raise ValueError(f"{case_id}: a missing image cannot be reviewed")
            if not row["reviewer_id"].strip():
                raise ValueError(f"{case_id}: reviewer_id is required")
            try:
                timestamp = datetime.fromisoformat(row["reviewed_at"].replace("Z", "+00:00"))
            except ValueError as error:
                raise ValueError(f"{case_id}: reviewed_at must be ISO-8601") from error
            if timestamp.tzinfo is None:
                raise ValueError(f"{case_id}: reviewed_at requires a timezone")
        if decision == "correct":
            status = row["corrected_status"]
            if status not in STATUSES:
                raise ValueError(f"{case_id}: corrected_status is invalid")
            point = row["corrected_destination_point_xy"].strip()
            box = row["corrected_destination_box_xyxy"].strip()
            question = row["corrected_clarification_question"].strip()
            if status == "proposed" and (not point or not box or question):
                raise ValueError(f"{case_id}: corrected proposal requires point + box and no question")
            if status == "needs_clarification" and (point or box or not question):
                raise ValueError(f"{case_id}: corrected clarification requires a question and no coordinates")
            if status == "insufficient_visual_evidence" and (point or box or question):
                raise ValueError(f"{case_id}: insufficient evidence requires empty corrections")
        elif any(row[field].strip() for field in CORRECTIONS):
            raise ValueError(f"{case_id}: correction fields are only allowed for decision=correct")
        result[case_id] = row
    return result


def corrected_target(row):
    if row["decision"] != "correct":
        return None
    point_text = row["corrected_destination_point_xy"].strip()
    box_text = row["corrected_destination_box_xyxy"].strip()
    question = row["corrected_clarification_question"].strip()
    try:
        point = json.loads(point_text) if point_text else None
        box = json.loads(box_text) if box_text else None
    except json.JSONDecodeError as error:
        raise ValueError(f"{row['case_id']}: corrected coordinates must be JSON arrays") from error
    return {
        "status": row["corrected_status"], "destination_point_xy": point,
        "destination_box_xyxy": box, "clarification_question": question or None,
    }


def outcome(row):
    correction = corrected_target(row)
    return row["decision"] if correction is None else "correct:" + json.dumps(correction, sort_keys=True)


def resolution_row(a, b):
    return {
        "case_id": a["case_id"], "reviewer_1_decision": a["decision"],
        "reviewer_2_decision": b["decision"], "reviewer_1_id": a["reviewer_id"],
        "reviewer_2_id": b["reviewer_id"], "reviewer_1_reviewed_at": a["reviewed_at"],
        "reviewer_2_reviewed_at": b["reviewed_at"],
        "reviewer_1_correction": json.dumps({f: a[f] for f in CORRECTIONS}, sort_keys=True),
        "reviewer_2_correction": json.dumps({f: b[f] for f in CORRECTIONS}, sort_keys=True),
        "resolution": "", "resolution_notes": "",
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESOLUTION_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer-1", type=Path, required=True)
    parser.add_argument("--reviewer-2", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--approved-cases", type=Path, default=Path(__file__).resolve().parents[1] / "review_cases.jsonl")
    parser.add_argument("--pending-cases", type=Path, default=Path(__file__).resolve().parents[1] / "reconfirmation_cases.jsonl")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    first, second = read_form(args.reviewer_1), read_form(args.reviewer_2)
    if set(first) != set(second):
        raise ValueError("Reviewer forms cover different case IDs")
    cases = {case["case_id"]: case for case in read_jsonl(args.approved_cases) + read_jsonl(args.pending_cases)}
    if set(cases) != set(first):
        raise ValueError("Reviewer forms and current case files cover different case IDs")

    resolved, disagreements, unavailable, pending, approved = [], [], [], [], []
    for case_id in sorted(first):
        a, b = first[case_id], second[case_id]
        case = cases[case_id]
        for reviewer in (a, b):
            for field, expected in (
                ("image_sha256", case["source"]["image_sha256"]),
                ("input_sha256", input_sha256(case)),
                ("supervision_sha256", supervision_sha256(case)),
                ("instruction", case["request"]["text"]),
                ("status", case["target"]["status"]),
            ):
                if reviewer[field] != expected:
                    raise ValueError(f"{case_id}: stale or edited {field} binding")
        if a["reviewer_id"] and b["reviewer_id"] and a["reviewer_id"] == b["reviewer_id"]:
            raise ValueError(f"{case_id}: reviewers must be different people")
        row = resolution_row(a, b)
        if not a["decision"] or not b["decision"]:
            row["resolution"] = "pending"
            pending.append(row)
            continue
        if a["decision"] == b["decision"] == "unavailable":
            row["resolution"] = "unavailable"
            unavailable.append(row)
            continue
        if outcome(a) != outcome(b):
            disagreements.append(row)
            continue
        row["resolution"] = outcome(a)
        resolved.append(row)
        if a["decision"] == "reject":
            continue
        result = deepcopy(case)
        if a["decision"] == "correct":
            result["target"] = corrected_target(a)
        result["target"] = normalized_target(result["target"], result["supervised_fields"], result["request"]["scene"]["frame"])
        result["review"] = {
            "status": "human_reviewed", "new_human_review": True,
            "approval_source": "two_independent_human_confirmations",
            "reviewers": [a["reviewer_id"], b["reviewer_id"]],
            "independent_reviewers_required": 2,
            "reviewed_at": b["reviewed_at"], "first_reviewed_at": a["reviewed_at"],
            "second_reviewed_at": b["reviewed_at"], "image_sha256": result["source"]["image_sha256"],
            "input_sha256": input_sha256(result), "supervision_sha256": supervision_sha256(result),
        }
        image_path = (root / result["request"]["scene"]["frame"]["image_path"]).resolve()
        result["request"]["scene"]["frame"]["image_path"] = Path(os.path.relpath(image_path, args.output_dir.resolve())).as_posix()
        approved.append(result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("resolved_reviews.csv", resolved), ("disagreements.csv", disagreements),
                       ("unavailable_cases.csv", unavailable), ("pending_cases.csv", pending)):
        write_csv(args.output_dir / name, rows)
    if approved:
        (args.output_dir / "approved_cases.jsonl").write_text(
            "".join(json.dumps(case, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for case in approved),
            encoding="utf-8",
        )
    status = {
        "status": "complete" if not (disagreements or pending or unavailable) else "reconfirmation_required",
        "current_cases": len(cases), "approved_cases_written": len(approved),
        "reviewed_cases": len(resolved), "pending_cases": len(pending),
        "unavailable_cases": len(unavailable), "unresolved_disagreements": len(disagreements),
        "rejected_cases": sum(row["resolution"] == "reject" for row in resolved),
        "approvals_applied_to_portable_jsonl": True,
    }
    (args.output_dir / "review_status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
