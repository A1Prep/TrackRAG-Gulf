"""Reproduce the six non-test passenger reviews from the corrected Task 2 Git handoff."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory

from trackrag.dropoff.data import load_cases, validate_disjoint_groups


REVISION = "2d96dcb6afdffb46c5d94d299768fe6a92acc094"
PACKAGE = "Task2_56_Case_Human_Review_Corrected"
PARTITIONS = ("train", "development")
RECORDS = {
    "train": {"local-train-video_054604-01051", "local-train-video_131333-00811",
              "local-train-video_142911-01501", "local-train-video_210906-01261"},
    "development": {"local-dev-video_054907-00781", "local-dev-video_174340-00781"},
}
PARTITION_FIELD = re.compile(rb'"partition"\s*:\s*"(train|development)"')


def git_blob(repo, path):
    return subprocess.run(["git", "-C", str(repo), "show", f"{REVISION}:{PACKAGE}/{path}"],
                          stdout=subprocess.PIPE, check=True).stdout


def materialize(repo, output):
    """Read approved nonfinal rows only. Copy their original JPEG bytes and verify the review bindings."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    approved = git_blob(repo, "review_cases.jsonl")
    ids = {p: set() for p in PARTITIONS}
    lines = {p: [] for p in PARTITIONS}
    images = {}
    with TemporaryDirectory(prefix=".review-import-", dir=output.parent) as temp:
        snapshot = Path(temp) / "snapshot"
        snapshot.mkdir()
        for raw in approved.splitlines():
            if not PARTITION_FIELD.search(raw):
                continue
            row = json.loads(raw)
            partition, case_id = row["partition"], row["case_id"]
            if case_id not in RECORDS.get(partition, ()) or case_id in ids[partition]:
                raise ValueError("Unexpected or duplicate nonfinal review")
            review = row["review"]
            if (review.get("approval_source") != "two_independent_human_confirmations"
                    or review.get("new_human_review") is not True or len(set(review["reviewers"])) != 2):
                raise ValueError("Two recorded reviewer confirmations are required")
            frame = row["request"]["scene"]["frame"]
            if not re.fullmatch(r"video_\d{6}", frame["recording_id"]):
                raise ValueError("Invalid reviewed recording identity")
            image = f"images/{partition}/{frame['recording_id']}/{frame['frame_id']:05d}.jpg"
            if image != frame["image_path"]:
                raise ValueError("Reviewed image must be in the corresponding nonfinal split")
            content = git_blob(repo, image)
            image_sha = hashlib.sha256(content).hexdigest()
            if image_sha != row["source"]["image_sha256"] or image_sha != review["image_sha256"]:
                raise ValueError("Reviewed image bytes changed")
            destination = snapshot / image
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            images[image] = image_sha
            ids[partition].add(case_id)
            lines[partition].append(raw)
        if ids != RECORDS:
            raise ValueError("Corrected nonfinal review set is incomplete")
        splits, stats = {}, {}
        for partition in PARTITIONS:
            path = snapshot / (partition + ".jsonl")
            path.write_bytes(b"\n".join(lines[partition]) + b"\n")
            rows = load_cases(path, allowed_partitions={partition})
            splits[partition] = rows
            stats[partition] = {"examples": len(rows), "recordings": len({r["scene_group"] for r in rows}),
                                "distinct_instructions": len({r["request"]["text"] for r in rows}),
                                "distinct_questions": len({r["target"]["clarification_question"] for r in rows}),
                                "reviewed_track_observations": sum(len(r["request"]["scene"]["tracks"]) for r in rows),
                                "statuses": dict(Counter(r["target"]["status"] for r in rows)),
                                "jsonl_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        validate_disjoint_groups(splits["train"], splits["development"])
        receipt = {"git_revision": REVISION, "git_package": PACKAGE,
                   "approved_blob_sha256": hashlib.sha256(approved).hexdigest(), "splits": stats,
                   "images": images, "reserved_rows_materialized": 0,
                   "review_description": "Two confirmations recorded in the corrected Task 2 review package. "
                                        "The six local examples repeat one instruction and one question across six scenes."}
        (snapshot / "source.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        snapshot.rename(output)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.repo, args.out), indent=2))


if __name__ == "__main__":
    main()
