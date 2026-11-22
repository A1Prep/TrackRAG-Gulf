"""Check a private offline context-review packet, then call its unchanged Task 2 resolver."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory
import unicodedata


RESOLVER_SHA256 = "3b903df45662d4f8442ab59396b28d0d6c6021a94442ee2db449c6128dda30cc"
FIXED = ("case_id", "partition", "recording_id", "frame_id", "image_available", "image_path", "image_sha256",
         "instruction", "status", "destination_point_xy", "destination_box_xyxy", "clarification_question",
         "input_sha256", "supervision_sha256")
LEGACY = ("legacy_decision", "legacy_corrected_destination_ref", "legacy_notes")


def require(condition, text):
    if not condition:
        raise ValueError(text)


def file_digest(path):
    return sha256(path.read_bytes()).hexdigest()


def reviewer_key(value):
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def resolve(packet, first, second, output):
    packet, output = Path(packet).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new output directory to keep earlier resolutions intact.")
    inventory = json.loads((packet / "packet-files.json").read_bytes())
    for name, expected in inventory.items():
        relative = PurePosixPath(name)
        path = (packet / relative).resolve()
        require(not relative.is_absolute() and path.is_relative_to(packet)
                and path.is_file() and file_digest(path) == expected, "Packet contents changed: " + name)
    require({"reviewer_1.csv", "reviewer_2.csv", "tools/dropoff_resolve_reviews.py", "reconfirmation_cases.jsonl",
             "review_cases.jsonl"}.issubset(inventory), "Incomplete packet inventory")
    resolver = packet / "tools/dropoff_resolve_reviews.py"
    require(file_digest(resolver) == RESOLVER_SHA256, "The original Task 2 resolver has changed")
    upstream = runpy.run_path(str(resolver))
    baseline = upstream["read_form"](packet / "reviewer_1.csv")
    require((packet / "reviewer_1.csv").read_bytes() == (packet / "reviewer_2.csv").read_bytes(),
            "Original blank forms differ")
    require(not (packet / "review_cases.jsonl").read_bytes(), "Input to this review must start fully pending")
    cases = upstream["read_jsonl"](packet / "reconfirmation_cases.jsonl")
    require(len(cases) == len(baseline) and {c["case_id"] for c in cases} == set(baseline),
            "Candidates and blank rows differ")
    for case in cases:
        template = baseline[case["case_id"]]
        image = (packet / template["image_path"]).resolve()
        require(image.is_relative_to(packet) and template["image_path"] in inventory
                and file_digest(image) == case["source"]["image_sha256"] == template["image_sha256"]
                and case["review"]["status"] == "pending_reconfirmation" and case["review"]["new_human_review"] is False
                and template["input_sha256"] == upstream["input_sha256"](case)
                and template["supervision_sha256"] == upstream["supervision_sha256"](case),
                "Photo, current context, or candidate binding differs")
        require(not any(template[k] for k in set(template) - set(FIXED)), "The source form already contains a return")
    # Freeze returned bytes before validation and pass exactly those bytes to the original resolver.
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as scratch:
        working = Path(scratch)
        form_reviewers = set()
        for role, origin in enumerate((first, second), 1):
            saved = working / f"return-{role}.csv"
            saved.write_bytes(Path(origin).read_bytes())
            returned = upstream["read_form"](saved)
            require(returned.keys() == baseline.keys(), "Returned form covers different cases")
            people = {reviewer_key(r["reviewer_id"]) for r in returned.values() if r["reviewer_id"].strip()}
            require(len(people) <= 1, "Each returned form must contain only one independent reviewer")
            require(not people.intersection(form_reviewers), "The two reviewer forms must come from different people")
            form_reviewers.update(people)
            for identity, row in returned.items():
                require(all(row[k] == baseline[identity][k] for k in FIXED) and not any(row[k] for k in LEGACY),
                        "Changed candidate metadata, photograph, binding or legacy field")
        # Resolver writes paths relative to its output. A sibling staging directory ensures portability.
        with TemporaryDirectory(dir=output.parent) as staging:
            sibling = Path(staging)
            resolved = subprocess.run([sys.executable, "-B", str(resolver), "--reviewer-1", str(working / "return-1.csv"),
                                       "--reviewer-2", str(working / "return-2.csv"), "--output-dir", str(sibling)],
                                      capture_output=True, text=True)
            require(resolved.returncode == 0, "Task 2 resolver: " + resolved.stderr.strip().split("\n")[-1])
            sibling.rename(output)
    return json.loads((output / "review_status.json").read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer-1", type=Path, required=True)
    parser.add_argument("--reviewer-2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(resolve(Path(__file__).resolve().parent, args.reviewer_1, args.reviewer_2, args.output),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
