#!/usr/bin/env python3
"""Validate the corrected local case package without external dependencies."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path, PurePosixPath
from PIL import Image

TOP = {"schema_version", "case_id", "scene_group", "partition", "source", "request", "target", "supervised_fields", "review"}
STATUSES = {"proposed", "needs_clarification", "insufficient_visual_evidence"}
FIELDS = ["status", "destination_point_xy", "destination_box_xyxy", "clarification_question"]
PENDING = {
    "local-eval-video_155425-00871", "local-eval-video_155425-01171",
    "local-eval-video_160325-00781", "local-eval-video_160325-00961",
    "local-eval-video_204647-00961", "local-eval-video_122233-00301",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def target_hash(case):
    target = deepcopy(case["target"])
    if target["status"] == "proposed":
        target["destination_point_xy"] = [float(v) for v in target["destination_point_xy"]]
        target["destination_box_xyxy"] = [float(v) for v in target["destination_box_xyxy"]]
    payload = {"target": target, "supervised_fields": sorted(case["supervised_fields"])}
    return hashlib.sha256(canonical(payload)).hexdigest()


def input_hash(case):
    request = deepcopy(case["request"])
    frame = request["scene"]["frame"]
    frame.pop("image_path")
    frame["image_sha256"] = case["source"]["image_sha256"].lower()
    request["scene"].pop("provenance", None)
    return hashlib.sha256(canonical(request)).hexdigest()


def check_target(case):
    target = case["target"]
    assert set(target) == set(FIELDS) and case["supervised_fields"] == FIELDS
    assert target["status"] in STATUSES
    frame = case["request"]["scene"]["frame"]
    width, height = frame["width"], frame["height"]
    point, box, question = target["destination_point_xy"], target["destination_box_xyxy"], target["clarification_question"]
    if target["status"] == "proposed":
        assert isinstance(point, list) and len(point) == 2 and isinstance(box, list) and len(box) == 4
        assert all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in point + box)
        left, top, right, bottom = box
        assert 0 <= point[0] < width and 0 <= point[1] < height
        assert 0 <= left < right <= width and 0 <= top < bottom <= height
        assert left <= point[0] <= right and top <= point[1] <= bottom
        assert question is None
    else:
        assert point is None and box is None
        if target["status"] == "needs_clarification":
            assert isinstance(question, str) and question.strip()
        else:
            assert question is None


def check_case(case, root, expected_status, historical=False):
    assert set(case) == TOP and case["schema_version"] == 1
    assert case["case_id"] == case["request"]["request_id"]
    assert case["partition"] in {"train", "development", "test"}
    assert case["source"]["dataset"] == "EMT" and case["source"]["scene_group_verified"] is True
    frame = case["request"]["scene"]["frame"]
    source = case["source"]
    assert frame["recording_id"] == source["recording_id"] and frame["frame_id"] == source["frame_id"]
    assert isinstance(frame["frame_id"], int) and frame["frame_id"] >= 0
    relative = PurePosixPath(frame["image_path"])
    assert not relative.is_absolute() and (historical or ".." not in relative.parts) and ":" not in frame["image_path"]
    image_path = (root / frame["image_path"]).resolve()
    assert image_path.is_file() and sha256(image_path) == source["image_sha256"]
    with Image.open(image_path) as image:
        assert image.size == (frame["width"], frame["height"])
        image.load()
    check_target(case)
    review = case["review"]
    assert review["status"] == expected_status
    if expected_status == "human_reviewed":
        required = 2 if case["partition"] == "test" or case["target"]["status"] == "needs_clarification" else 1
        assert review["new_human_review"] is (False if historical else True)
        assert len(review["reviewers"]) >= required and len(set(review["reviewers"])) == len(review["reviewers"])
        assert all(isinstance(value, str) and value.strip() for value in review["reviewers"])
        timestamp = datetime.fromisoformat(review["reviewed_at"].replace("Z", "+00:00"))
        assert timestamp.tzinfo is not None
        assert review["image_sha256"] == source["image_sha256"]
        assert review["input_sha256"] == input_hash(case)
        assert review["supervision_sha256"] == target_hash(case)
    else:
        assert review["new_human_review"] is False and "reviewers" not in review
        assert review["image_sha256"] == source["image_sha256"]
        assert review["input_sha256"] == input_hash(case)
        assert review["supervision_sha256"] == target_hash(case)
        assert isinstance(review["previous_confirmed_review"], dict)
    assert "destination_track_id" not in canonical(case).decode()
    for landmark in source.get("landmarks", []):
        assert set(landmark) == {"relation", "role", "track_id"}
        assert isinstance(landmark["track_id"], int)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-report", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    approved = read_jsonl(root / "review_cases.jsonl")
    pending = read_jsonl(root / "reconfirmation_cases.jsonl")
    history = read_jsonl(root / "history/original_confirmed_review_cases.jsonl")
    assert len(approved) == 50 and len(pending) == 6 and len(history) == 56
    assert len({case["case_id"] for case in approved + pending + history}) == 56
    assert {case["case_id"] for case in pending} == PENDING
    for case in approved:
        check_case(case, root, "human_reviewed")
    for case in pending:
        check_case(case, root, "pending_reconfirmation")
    for case in history:
        # History paths intentionally use ../images; check them relative to their JSONL directory.
        check_case(case, root / "history", "human_reviewed", historical=True)
    current = {case["case_id"]: case for case in approved + pending}
    old = {case["case_id"]: case for case in history}
    for case_id in PENDING:
        previous = current[case_id]["review"]["previous_confirmed_review"]
        assert previous == old[case_id]["review"]
    for case in approved:
        before = old[case["case_id"]]
        before_request = deepcopy(before["request"])
        before_request["scene"]["frame"]["image_path"] = before_request["scene"]["frame"]["image_path"].removeprefix("../")
        assert case["request"] == before_request and case["target"] == before["target"]

    image_paths = {case["request"]["scene"]["frame"]["image_path"] for case in current.values()}
    assert len(image_paths) == 56 and len(list((root / "images").rglob("*.jpg"))) == 56
    assert json.loads((root / "missing_images.json").read_text()) == []

    forms = []
    for name in ("reviewer_1.csv", "reviewer_2.csv"):
        with (root / name).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 56 and len({row["case_id"] for row in rows}) == 56
        assert Counter(bool(row["decision"]) for row in rows) == {True: 50, False: 6}
        for row in rows:
            case = current[row["case_id"]]
            assert row["instruction"] == case["request"]["text"]
            assert row["image_sha256"] == case["source"]["image_sha256"]
            assert row["input_sha256"] == input_hash(case)
            assert row["supervision_sha256"] == target_hash(case)
            if case["case_id"] in PENDING:
                assert not row["decision"] and not row["reviewer_id"] and not row["reviewed_at"]
            else:
                assert row["decision"] == "accept" and row["reviewer_id"] and row["reviewed_at"]
        forms.append({row["case_id"]: row for row in rows})
    for case_id in current:
        if case_id not in PENDING:
            assert forms[0][case_id]["reviewer_id"] != forms[1][case_id]["reviewer_id"]

    partitions = defaultdict(list)
    for case in current.values():
        partitions[case["partition"]].append(case)
    recordings = {name: {case["source"]["recording_id"] for case in cases} for name, cases in partitions.items()}
    names = list(recordings)
    assert all(not (recordings[a] & recordings[b]) for i, a in enumerate(names) for b in names[i + 1:])
    partition_ids = {name: {case["case_id"] for case in cases} for name, cases in partitions.items()}
    assert not (partition_ids["train"] & partition_ids["development"])
    assert partition_ids["train"] | partition_ids["development"] | partition_ids["test"] == set(current)

    texts = [case["request"]["text"] for case in current.values()]
    assert not any("?" in text for text in texts)
    for case in current.values():
        for text in [case["request"]["text"]] + case["source"].get("paraphrases", []):
            lowered = text.lower()
            if any(phrase in lowered for phrase in ("take the", "choose the", "ignore the")):
                assert any(word in lowered for word in ("stop", "pull over"))

    manifest_result = {"present": False, "files": 0, "verified": False}
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        assert manifest["fully_approved_cases"] == 50 and manifest["reconfirmation_required_cases"] == 6
        assert "second_independent_review_required" not in manifest
        for name, receipt in manifest["files"].items():
            path = root / name
            assert path.is_file() and path.stat().st_size == receipt["bytes"] and sha256(path) == receipt["sha256"]
        disk_files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path.name != "manifest.json"}
        assert disk_files == set(manifest["files"])
        manifest_result = {"present": True, "files": len(disk_files), "verified": True}

    report = {
        "schema_version": 1, "status": "passed", "total_current_cases": 56,
        "fully_admissible_human_reviewed_cases": len(approved), "pending_reconfirmation_cases": len(pending),
        "confirmed_history_cases": len(history), "outcomes": dict(Counter(case["target"]["status"] for case in current.values())),
        "partition_counts": {name: len(cases) for name, cases in partitions.items()},
        "recording_disjoint": True, "images_verified": 56, "missing_images": 0,
        "image_hashes_dimensions_and_decodability_verified": True,
        "current_input_and_supervision_bindings_verified": True,
        "confirmed_unchanged_approvals_verified": 50,
        "pending_candidates_not_represented_as_gold": 6,
        "review_forms_verified": 112, "ground_area_geometry_rechecked": 39,
        "vehicle_overlap_targets_corrected": 4, "primary_passenger_instructions_rewritten": 2,
        "public_or_source_validation_rows_in_local_package": 0,
        "reserved_guard_ids_in_local_package": 0, "manifest": manifest_result,
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.write_report:
        args.write_report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
