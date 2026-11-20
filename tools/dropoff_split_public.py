"""Split verified public training annotations for a destination-only pilot.

No final-test or clarification labels are created. Source images and masks stay
in their original directory; exported paths are relative to the split folder.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

from trackrag.dropoff.contracts import DESTINATION_FIELDS
from trackrag.dropoff.data import (
    inference_input_sha256, load_cases, supervision_sha256, validate_disjoint_groups,
)


MAX_CASE_BYTES = 128 * 1024 * 1024


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _read_public(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        payload = stream.read(MAX_CASE_BYTES + 1)
    if len(payload) > MAX_CASE_BYTES:
        raise ValueError("Public case file exceeds the snapshot byte budget")
    rows = load_cases(path, allowed_partitions={"train"}, payload=payload)
    for row in rows:
        source, review = row["source"], row["review"]
        if (source.get("dataset") != "Talk2Car-RegSeg"
                or source.get("source_split") != "train"
                or source.get("scene_group_verified") is not True
                or review.get("status") != "public_annotation"
                or ("new_human_review" in review and review["new_human_review"] is not False)
                or set(row["supervised_fields"]) != DESTINATION_FIELDS
                or not source.get("mask_path")):
            raise ValueError("Only verified public RegSeg train masks with destination-only labels are supported")
    return path, payload, rows


def _identity(row):
    frame = row["request"]["scene"]["frame"]
    return (row["scene_group"], row["source"]["source_index"],
            row["source"]["image_sha256"], row["source"]["mask_sha256"],
            inference_input_sha256(row["request"], row["source"]["image_sha256"]),
            supervision_sha256(row["target"], row["supervised_fields"], frame))


def _relative(path, parent):
    try:
        return Path(os.path.relpath(path, parent)).as_posix()
    except ValueError as error:
        raise ValueError("Split output and source artifacts must share a filesystem volume") from error


def _export_row(row, partition, output):
    row = deepcopy(row)
    row["partition"] = partition
    frame = row["request"]["scene"]["frame"]
    frame["image_path"] = _relative(frame["image_path"], output)
    row["source"]["mask_path"] = _relative(row["source"]["mask_path"], output)
    return row


def split_public(cases_path, output, *, development_cases=20, seed=42, keep_training_cases=None):
    """Select whole source scenes, retaining previously used pilot scenes in train."""
    if type(development_cases) is not int or development_cases < 1:
        raise ValueError("development_cases must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError("Choose a new split output directory")
    source_path, source_payload, rows = _read_public(cases_path)
    if development_cases >= len(rows):
        raise ValueError("Development target must leave at least one training case")
    keep_path, keep_payload, keep = None, None, []
    if keep_training_cases is not None:
        keep_path, keep_payload, keep = _read_public(keep_training_cases)

    identities, image_scenes = {}, {}
    for row in rows + keep:
        case_id = row["case_id"]
        identity = _identity(row)
        if case_id in identities and identities[case_id] != identity:
            raise ValueError("A repeated pilot case identity has changed source or supervision")
        identities[case_id] = identity
        image_hash = row["source"]["image_sha256"]
        previous = image_scenes.setdefault(image_hash, row["scene_group"])
        if previous != row["scene_group"]:
            raise ValueError("Identical image bytes are assigned to different source scenes")

    groups = defaultdict(list)
    for row in rows:
        groups[row["scene_group"]].append(row)
    locked = {row["scene_group"] for row in keep}
    candidates = sorted(
        (group for group in groups if group not in locked),
        key=lambda group: (_sha(f"{seed}:{group}".encode("utf-8")), group),
    )
    if not candidates:
        raise ValueError("No source scene remains eligible for development")
    chosen, count = set(), 0
    for group in candidates:
        size = len(groups[group])
        if count + size <= development_cases and count + size < len(rows):
            chosen.add(group)
            count += size
    # Whole groups can prevent an exact target. Keep both partitions nonempty
    # and report the actual count rather than splitting an adjacent scene.
    if not chosen:
        eligible = [group for group in candidates if len(groups[group]) < len(rows)]
        if not eligible:
            raise ValueError("At least two independent source scenes are needed")
        chosen.add(min(eligible, key=lambda group: abs(len(groups[group]) - development_cases)))
    train = sorted((row for row in rows if row["scene_group"] not in chosen), key=lambda row: row["case_id"])
    development = sorted((row for row in rows if row["scene_group"] in chosen), key=lambda row: row["case_id"])
    if not train or not development:
        raise ValueError("Both training and development must be nonempty")
    validate_disjoint_groups(train, development)
    # Previously used scenes and images must not enter checkpoint selection.
    validate_disjoint_groups(keep, development)
    exported = {
        "train.jsonl": [_export_row(row, "train", output) for row in train],
        "development.jsonl": [_export_row(row, "development", output) for row in development],
    }
    payloads = {name: "".join(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
                             for row in values).encode("utf-8") for name, values in exported.items()}
    # Revalidate portable paths and the changed partition before creating output.
    for name, partition in (("train.jsonl", "train"), ("development.jsonl", "development")):
        load_cases(output / name, allowed_partitions={partition}, payload=payloads[name])
    manifest = {
        "schema_version": 1, "kind": "public_destination_only_pilot_splits",
        "source_cases_path": _relative(source_path, output), "source_cases_sha256": _sha(source_payload),
        "keep_training_cases_path": _relative(keep_path, output) if keep_path else None,
        "keep_training_cases_sha256": _sha(keep_payload) if keep_payload is not None else None,
        "seed": seed, "selection_rule": "seeded_sha256_scene_order_then_whole_groups_up_to_target",
        "requested_development_cases": development_cases, "train_cases": len(train),
        "development_cases": len(development), "source_scene_count": len(groups),
        "train_case_ids": [row["case_id"] for row in train],
        "development_case_ids": [row["case_id"] for row in development],
        "train_scene_groups": sorted({row["scene_group"] for row in train}),
        "development_scene_groups": sorted(chosen),
        "kept_training_scene_groups": sorted(locked & groups.keys()),
        "files": {name: {"sha256": _sha(payload), "bytes": len(payload)} for name, payload in payloads.items()},
        "supervised_fields": sorted(DESTINATION_FIELDS),
        "new_human_review": False, "final_evaluation_set": False,
        "warning": "Public training-source destination geometry only. Development is for pilot selection, not final EMT, clarification, interaction or voice evaluation.",
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        (output / name).write_bytes(payload)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--development-cases", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-training-cases", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = split_public(args.cases, args.out, development_cases=args.development_cases,
                                seed=args.seed, keep_training_cases=args.keep_training_cases)
    except (OSError, ValueError) as error:
        print(f"Public pilot split failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
