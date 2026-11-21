"""Task 2 instruction-to-destination data preparation and validation.

Public RegSeg masks are navigable-region supervision, not automatically drop-off
masks, anchor identities, ambiguity labels, or dialogue responses. Public cases
are therefore tagged ``rnr_region`` and assigned to train only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

LOCAL_HEADER = [
    "recording_id", "frame_id", "destination_ref", "instruction",
    "paraphrases", "tags", "landmarks", "parent_case_id",
]
LOCAL_TAGS = {
    "unique_destination", "two_landmark", "ambiguous", "missing_reference",
    "occluded_landmark", "correction", "contrasting",
}
PUBLIC_AUDIT_COUNTS = {"train": 170, "val": 179}


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_regseg_annotations(path: Path | str, source_split: str) -> list[dict[str, Any]]:
    """Parse ``image_id instruction`` lines without treating source val as dev."""
    if source_split not in {"train", "val"}:
        raise ValueError("source_split must be train or val")
    path = Path(path)
    cases = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip():
            continue
        match = re.match(r"^(\d+)\s+(.+?)\s*$", raw)
        if not match:
            raise ValueError(f"{path}:{line_number}: expected 'image_id instruction'")
        image_id = int(match.group(1))
        cases.append({
            "case_id": f"regseg_{source_split}_{image_id}",
            "source": "talk2car_regseg",
            "source_split": source_split,
            "image_id": image_id,
            "split": "train",
            "scene_id": None,
            "instruction": match.group(2),
            "difficulty_tags": ["rnr_region"],
            "image_path": f"imgs/img_{source_split}_{image_id}.jpg",
            "mask_path": f"val_masks_new/gt_img_ann_{source_split}_{image_id}.png",
        })
    ids = [case["case_id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate image IDs")
    return cases


def mask_target_and_box(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, int, int, int], float]:
    """Return an interior pixel, half-open coarse box, and interior clearance.

    The target maximizes distance to the mask background; ties are resolved by
    NumPy's first flattened index. The box is [xmin, ymin, xmax+1, ymax+1].
    """
    if mask.ndim != 2:
        raise ValueError("Mask must be a two-dimensional array")
    binary = (mask > 0).astype(np.uint8)
    if not binary.any():
        raise ValueError("Mask has no foreground pixels")
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
    if binary[y, x] == 0:
        raise AssertionError("Derived target is outside the mask")
    ys, xs = np.nonzero(binary)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return (int(x), int(y)), box, float(distance[y, x])


def validate_public_case(case: dict[str, Any], data_root: Path | str) -> dict[str, Any]:
    """Join actual pixels and return a record with mask-derived supervision."""
    root = Path(data_root)
    image_path = root / case["image_path"]
    mask_path = root / case["mask_path"]
    try:
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Cannot decode image: {image_path}") from exc
    try:
        with Image.open(mask_path) as mask_image:
            mask_image.load()
            mask = np.asarray(mask_image.convert("L"))
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Cannot decode mask: {mask_path}") from exc
    if mask.shape != (height, width):
        raise ValueError(
            f"Image/mask dimensions differ: image={width}x{height}, "
            f"mask={mask.shape[1]}x{mask.shape[0]}"
        )
    target, box, clearance = mask_target_and_box(mask)
    result = dict(case)
    result.update({
        "width": width,
        "height": height,
        "mask_area_px": int(np.count_nonzero(mask)),
        "target_xy": list(target),
        "target_clearance_px": clearance,
        "coarse_box_xyxy": list(box),
        "target_rule": "max_cv2_distance_transform_l2_first_flattened_argmax",
        "box_rule": "foreground_xyxy_half_open_max_plus_one",
        "pixel_validated": True,
    })
    return result


def model_input(case: dict[str, Any]) -> dict[str, Any]:
    """Export model inputs separately from all gold/target fields."""
    return {
        "case_id": case["case_id"],
        "image_path": case["image_path"],
        "instruction": case["instruction"],
    }


def invalid_receipt(case: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "case_id": case.get("case_id"),
        "source": case.get("source"),
        "image_path": case.get("image_path"),
        "mask_path": case.get("mask_path"),
        "source_file": case.get("source_file"),
        "source_row": case.get("source_row"),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def prepare_public_cases(
    data_root: Path | str,
    annotation_dir: Path | str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Validate every available public case; never pad to a scheduling target."""
    annotation_dir = Path(annotation_dir)
    parsed = []
    observed = {}
    for source_split in ("train", "val"):
        path = annotation_dir / f"annotation_new_{source_split}.txt"
        split_cases = parse_regseg_annotations(path, source_split)
        observed[source_split] = len(split_cases)
        parsed.extend(split_cases)
    valid, excluded = [], []
    for case in parsed:
        try:
            valid.append(validate_public_case(case, data_root))
        except (OSError, ValueError) as exc:
            excluded.append(invalid_receipt(case, exc))
    summary = {
        "source_observed_counts": observed,
        "source_observed_total": len(parsed),
        "supplied_audit_counts": PUBLIC_AUDIT_COUNTS,
        "supplied_audit_total": sum(PUBLIC_AUDIT_COUNTS.values()),
        "audit_matches_source": observed == PUBLIC_AUDIT_COUNTS,
        "valid_count": len(valid),
        "excluded_count": len(excluded),
        "scene_mapping_available": False,
        "assigned_split": "train",
        "reserved_scene_check": "not_enforceable_without_image_id_to_scene_id",
        "difficulty_tag": "rnr_region",
        "scheduling_target": "500-1000",
        "target_policy": "document_shortfall_or_surplus_without_padding",
    }
    return valid, excluded, summary


def parse_landmarks(raw: str) -> list[dict[str, Any]]:
    landmarks = []
    if not raw:
        return landmarks
    for item in raw.split(";"):
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid landmark: {item!r}")
        track_id, relation, role = parts
        if relation not in {"left", "right", "front"} or role not in {"primary", "secondary"}:
            raise ValueError(f"Invalid landmark relation/role: {item!r}")
        landmarks.append({"track_id": int(track_id), "relation": relation, "role": role})
    return landmarks


def _local_image_path(
    emt_root: Path | None, split: str, recording_id: str, frame_id: int,
) -> str:
    disk_split = "train" if split in {"train", "dev"} else "test"
    relative = Path("images") / disk_split / recording_id / f"{frame_id:05d}.jpg"
    if emt_root is None:
        return relative.as_posix()
    image_dir = emt_root / "images" / disk_split / recording_id
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Recording image folder missing: {image_dir}")
    matches = [
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and path.stem.isdigit() and int(path.stem) == frame_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one frame {frame_id} in {image_dir}; found {len(matches)}")
    with Image.open(matches[0]) as image:
        image.load()
        width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {matches[0]}")
    return str(matches[0].resolve())


def read_local_cases(
    annotation_root: Path | str,
    emt_root: Path | str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read local CSVs and keep gold references separate from model inputs."""
    annotation_root = Path(annotation_root)
    root = Path(emt_root) if emt_root is not None else None
    valid, excluded = [], []
    for split in ("train", "dev", "eval"):
        for path in sorted((annotation_root / split).glob("*.csv")):
            with path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != LOCAL_HEADER:
                    raise ValueError(f"{path}: header must be exactly {LOCAL_HEADER}")
                for row_number, row in enumerate(reader, 2):
                    case = {
                        "case_id": "",
                        "source": "local_emt",
                        "split": split,
                        "source_file": path.name,
                        "source_row": row_number,
                    }
                    try:
                        recording_id = row["recording_id"].strip()
                        frame_id = int(row["frame_id"])
                        instruction = row["instruction"].strip()
                        paraphrases = [x.strip() for x in row["paraphrases"].split("|")]
                        tag = row["tags"].strip()
                        destination = row["destination_ref"].strip()
                        if not recording_id or not instruction:
                            raise ValueError("recording_id and instruction are required")
                        if len(paraphrases) != 2 or not all(paraphrases):
                            raise ValueError("Exactly two nonempty paraphrases are required")
                        if tag not in LOCAL_TAGS:
                            raise ValueError(f"Unknown tag: {tag}")
                        if not re.fullmatch(r"track:\d+", destination):
                            raise ValueError("destination_ref must be track:<track_id>")
                        landmarks = parse_landmarks(row["landmarks"].strip())
                        destination_track = int(destination.split(":")[1])
                        if tag in {"two_landmark", "occluded_landmark"} and len(landmarks) != 2:
                            raise ValueError(f"{tag} requires two landmarks")
                        if tag in {"correction", "contrasting"} and not landmarks:
                            raise ValueError(f"{tag} requires at least one landmark")
                        if tag == "ambiguous" and landmarks:
                            raise ValueError("ambiguous cases use no landmarks")
                        case.update({
                            "case_id": f"local_{split}_{recording_id}_{frame_id:05d}",
                            "recording_id": recording_id,
                            "frame_id": frame_id,
                            "source_group_id": recording_id,
                            "image_path": _local_image_path(root, split, recording_id, frame_id),
                            "instruction": instruction,
                            "paraphrases": paraphrases,
                            "difficulty_tags": [tag],
                            "destination_track_id": destination_track,
                            "landmarks": landmarks,
                            "parent_case_id": row["parent_case_id"].strip() or None,
                            "review_status": "pending_independent_human_review",
                        })
                        valid.append(case)
                    except (KeyError, OSError, ValueError) as exc:
                        excluded.append(invalid_receipt(case, exc))
    return valid, excluded


def assert_local_split_guards(cases: Iterable[dict[str, Any]]) -> None:
    """Enforce recording-level separation and globally unique case IDs."""
    owner: dict[str, str] = {}
    case_ids = set()
    for case in cases:
        if case["source"] != "local_emt":
            continue
        recording = case["recording_id"]
        split = case["split"]
        if recording in owner and owner[recording] != split:
            raise ValueError(f"Recording {recording} occurs in {owner[recording]} and {split}")
        owner[recording] = split
        if case["case_id"] in case_ids:
            raise ValueError(f"Duplicate case ID: {case['case_id']}")
        case_ids.add(case["case_id"])


def reserved_scene_guard(public_cases: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Report, rather than fake, scene-disjointness when no scene map exists."""
    cases = list(public_cases)
    mapped = {case.get("scene_id") for case in cases if case.get("scene_id") is not None}
    if not mapped:
        return {
            "status": "not_enforceable",
            "reason": "RegSeg supplies no image_id to scene_id mapping",
            "unique_scenes_summary_value": 1,
            "scene_mapping_available": False,
            "mitigation": "public_train_only_local_dev_eval",
        }
    raise NotImplementedError("Reserved-scene verification requires the pinned 58-scene audit")


def split_counts(cases: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(case["split"] for case in cases)
    return {split: counts.get(split, 0) for split in ("train", "dev", "eval")}


def write_json(path: Path | str, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_release(
    output_dir: Path | str,
    local_cases: list[dict[str, Any]],
    local_exclusions: list[dict[str, Any]],
    public_cases: list[dict[str, Any]] | None = None,
    public_exclusions: list[dict[str, Any]] | None = None,
    public_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write versioned manifests, model inputs, targets, and review handoffs."""
    output_dir = Path(output_dir)
    public_cases = public_cases or []
    public_exclusions = public_exclusions or []
    assert_local_split_guards(local_cases)
    all_cases = public_cases + local_cases
    all_ids = [case["case_id"] for case in all_cases]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Duplicate case IDs across public and local releases")
    if any(case["split"] != "train" for case in public_cases):
        raise ValueError("Public RegSeg cases must be train-only")

    manifest = {
        "schema_version": "dropoff-1.0",
        "release_status": "draft_pending_independent_human_review",
        "counts": split_counts(all_cases),
        "local_counts": split_counts(local_cases),
        "public_valid_count": len(public_cases),
        "public_excluded_count": len(public_exclusions),
        "local_excluded_count": len(local_exclusions),
        "public_summary": public_summary,
        "scene_guard": reserved_scene_guard(public_cases),
        "review_requirement": {
            "eval_cases": 2,
            "ambiguous_cases": 2,
            "reviewers_must_be_independent_humans": True,
            "self_authoring_is_not_independent_review": True,
        },
        "gold_field_policy": {
            "public": ["mask_path", "target_xy", "coarse_box_xyxy"],
            "local": ["destination_track_id", "landmarks"],
            "not_automatically_labelled": [
                "anchor_identities", "ambiguity", "dialogue_responses", "scene_id"
            ],
        },
        "cases": all_cases,
    }
    write_json(output_dir / "splits" / "dropoff_manifest.json", manifest)
    write_json(output_dir / "annotations" / "dropoff" / "public_exclusions.json", public_exclusions)
    write_json(output_dir / "annotations" / "dropoff" / "local_exclusions.json", local_exclusions)

    inputs_dir = output_dir / "annotations" / "dropoff" / "model_inputs"
    targets_dir = output_dir / "annotations" / "dropoff" / "targets"
    for split in ("train", "dev", "eval"):
        split_cases = [case for case in all_cases if case["split"] == split]
        write_json(inputs_dir / f"{split}.json", [model_input(case) for case in split_cases])
        gold = []
        for case in split_cases:
            target = {
                "case_id": case["case_id"],
                "source": case["source"],
                "difficulty_tags": case["difficulty_tags"],
            }
            for key in (
                "mask_path", "target_xy", "target_clearance_px", "coarse_box_xyxy",
                "destination_track_id", "landmarks",
            ):
                if key in case:
                    target[key] = case[key]
            gold.append(target)
        write_json(targets_dir / f"{split}.json", gold)

    review_cases = [
        case for case in local_cases
        if case["split"] == "eval" or "ambiguous" in case["difficulty_tags"]
    ]
    review_header = [
        "case_id", "split", "recording_id", "frame_id", "instruction",
        "paraphrases", "decision", "corrected_destination_ref", "notes",
    ]
    for reviewer in (1, 2):
        path = output_dir / "reviews" / f"reviewer_{reviewer}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=review_header)
            writer.writeheader()
            for case in review_cases:
                writer.writerow({
                    "case_id": case["case_id"], "split": case["split"],
                    "recording_id": case["recording_id"], "frame_id": case["frame_id"],
                    "instruction": case["instruction"],
                    "paraphrases": "|".join(case["paraphrases"]),
                    "decision": "", "corrected_destination_ref": "", "notes": "",
                })
    write_json(output_dir / "reviews" / "review_status.json", {
        "status": "pending",
        "required_independent_human_reviews": 2,
        "cases_requiring_review": len(review_cases),
        "completed_reviews": 0,
        "disagreements_resolved": 0,
    })
    return manifest
