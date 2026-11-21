#!/usr/bin/env python3
"""Merge two independent human review forms and expose unresolved disagreements."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ALLOWED = {"accept", "reject", "correct"}


def read_form(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        case_id = row["case_id"]
        if case_id in result:
            raise ValueError(f"Duplicate review for {case_id}")
        decision = row["decision"].strip().lower()
        if decision not in ALLOWED:
            raise ValueError(f"{case_id}: decision must be accept, reject, or correct")
        correction = row["corrected_destination_ref"].strip()
        if decision == "correct" and not correction.startswith("track:"):
            raise ValueError(f"{case_id}: correction requires track:<track_id>")
        row["decision"] = decision
        row["corrected_destination_ref"] = correction
        result[case_id] = row
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer-1", type=Path, required=True)
    parser.add_argument("--reviewer-2", type=Path, required=True)
    parser.add_argument("--reviewer-1-id", required=True)
    parser.add_argument("--reviewer-2-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.reviewer_1_id == args.reviewer_2_id:
        raise ValueError("Reviewers must be different people")
    first, second = read_form(args.reviewer_1), read_form(args.reviewer_2)
    if set(first) != set(second):
        raise ValueError("Reviewer forms cover different case IDs")

    resolved, disagreements = [], []
    for case_id in sorted(first):
        a, b = first[case_id], second[case_id]
        outcome_a = a["corrected_destination_ref"] if a["decision"] == "correct" else a["decision"]
        outcome_b = b["corrected_destination_ref"] if b["decision"] == "correct" else b["decision"]
        row = {
            "case_id": case_id, "reviewer_1_decision": a["decision"],
            "reviewer_2_decision": b["decision"],
            "reviewer_1_correction": a["corrected_destination_ref"],
            "reviewer_2_correction": b["corrected_destination_ref"],
            "resolution": "", "resolver_id": "", "resolution_notes": "",
        }
        if outcome_a == outcome_b:
            row["resolution"] = outcome_a
            row["resolver_id"] = "agreement"
            resolved.append(row)
        else:
            disagreements.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    header = list(resolved[0] if resolved else disagreements[0])
    for name, rows in (("resolved_reviews.csv", resolved), ("disagreements.csv", disagreements)):
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
    status = {
        "status": "complete" if not disagreements else "disagreements_pending",
        "reviewer_1_id": args.reviewer_1_id,
        "reviewer_2_id": args.reviewer_2_id,
        "independent_human_attestation_required": True,
        "reviewed_cases": len(first),
        "agreements": len(resolved),
        "unresolved_disagreements": len(disagreements),
    }
    (args.output_dir / "review_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
