"""Versioned training cases with partial labels and explicit partition checks."""

from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from .contracts import DESTINATION_FIELDS, validate_request, validate_target


REGSEG_GUARDS_SHA256 = "27a218f0f71c8bae59f9c31f603e4a531e687c7ee42596d359e2e466d87fadfb"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(value, base):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _verify_hash(path, expected, label):
    actual = file_sha256(path)
    if not isinstance(expected, str) or len(expected) != 64 or actual != expected.lower():
        raise ValueError(f"{label} SHA256 is missing or does not match the declared source")
    return actual


def supervision_sha256(target, supervised_fields, frame):
    normalized = validate_target(target, supervised_fields, frame)
    payload = {"target": normalized, "supervised_fields": sorted(supervised_fields)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def inference_input_sha256(request, image_sha256):
    """Portable review binding to the interpreted input, excluding local paths."""
    value = validate_request(request)
    frame = value["scene"]["frame"]
    frame.pop("image_path")
    frame["image_sha256"] = image_sha256.lower() if isinstance(image_sha256, str) else image_sha256
    value["scene"].pop("provenance", None)  # This metadata is not sent to the model.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _regseg_group_guard(case):
    source = case["source"]
    if source.get("dataset") != "Talk2Car-RegSeg":
        return
    path = Path(__file__).resolve().parents[2] / "configs/dropoff/regseg-scene-guards.json"
    if not path.is_file() or file_sha256(path) != REGSEG_GUARDS_SHA256:
        raise ValueError("pinned RegSeg scene guard is unavailable or changed")
    rules = json.loads(path.read_text(encoding="utf-8"))
    if source.get("repository_revision") != rules["regseg_revision"] or source.get("metadata_sha256") != rules["regseg_metadata_sha256"]:
        raise ValueError("RegSeg case is not bound to the audited metadata revision")
    index = source.get("source_index")
    if type(index) is not int or not 0 <= index < len(rules["train_scene_indices"]):
        raise ValueError("RegSeg training source_index is not valid")
    scene = rules["scene_tokens"][rules["train_scene_indices"][index]]
    if source.get("scene_token") != scene or case["scene_group"] != "talk2car-scene:" + scene:
        raise ValueError("RegSeg scene group does not match the audited source-index mapping")
    if case["partition"] in ("train", "development") and (
        scene in rules["reserved_scene_tokens"] or index in rules["concise_test_train_ids"]
    ):
        raise ValueError("RegSeg source scene is reserved for concise-test evaluation")
    if case["review"]["status"] == "public_annotation":
        instruction_sha256 = hashlib.sha256(case["request"]["text"].encode("utf-8")).hexdigest()
        if instruction_sha256 != rules["train_command_sha256"][index]:
            raise ValueError("public RegSeg instruction differs from its audited source command")


def _human_review_guard(case):
    review = case["review"]
    if review["status"] != "human_reviewed":
        return
    reviewers = review.get("reviewers")
    required = 2 if case["partition"] == "test" or case["target"].get("status") == "needs_clarification" else 1
    if not isinstance(reviewers, list) or not all(isinstance(x, str) and x.strip() for x in reviewers):
        raise ValueError("human review requires reviewer identities")
    if len(set(reviewers)) != len(reviewers) or len(reviewers) < required:
        raise ValueError("independent human reviewer coverage is incomplete")
    reviewed_at = review.get("reviewed_at")
    if not isinstance(reviewed_at, str):
        raise ValueError("human review requires a valid review timestamp")
    try:
        timestamp = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("human review requires a valid review timestamp") from error
    if timestamp.tzinfo is None:
        raise ValueError("human review timestamp requires a timezone")
    if review.get("image_sha256") != case["source"].get("image_sha256"):
        raise ValueError("human review is not bound to the inspected image")
    expected = supervision_sha256(case["target"], case["supervised_fields"],
                                  case["request"]["scene"]["frame"])
    if review.get("supervision_sha256") != expected:
        raise ValueError("human review is not bound to the current supervised labels")
    expected_input = inference_input_sha256(case["request"], case["source"].get("image_sha256"))
    if review.get("input_sha256") != expected_input:
        raise ValueError("human review is not bound to the current instruction and scene")


def validate_case(case, *, base_dir=None, allowed_partitions=("train",), verify_images=True):
    """Load reviewed labels without converting unknown fields into false negatives."""
    if not isinstance(case, dict):
        raise ValueError("case must be an object")
    try:
        json.dumps(case, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("case must contain only finite JSON values") from error
    required = {"schema_version", "case_id", "scene_group", "partition", "source",
                "request", "target", "supervised_fields", "review"}
    if set(case) != required:
        raise ValueError(f"case must have exactly the versioned fields: {sorted(required)}")
    if type(case["schema_version"]) is not int or case["schema_version"] != 1:
        raise ValueError("unsupported case schema version")
    for field in ("case_id", "scene_group"):
        if not isinstance(case[field], str) or not case[field].strip() or len(case[field]) > 300:
            raise ValueError(f"{field} must be nonempty bounded text")
    if case["partition"] not in ("train", "development", "test"):
        raise ValueError("unknown data partition")
    if case["partition"] not in set(allowed_partitions):
        raise ValueError(f"partition {case['partition']} is not allowed in this input")
    if not isinstance(case["source"], dict) or not isinstance(case["review"], dict):
        raise ValueError("case source/review must be objects")
    if case["review"].get("status") not in ("public_annotation", "human_reviewed"):
        raise ValueError("pending, uncertain and AI-only labels are not training/reference gold")
    if not isinstance(case["source"].get("dataset"), str) or not case["source"]["dataset"].strip():
        raise ValueError("source dataset is required")
    if case["source"].get("source_split") in ("test", "validation", "val") and case["partition"] == "train":
        raise ValueError("public validation/test source cannot become training data")
    selecting = case["partition"] in ("train", "development")
    if selecting and case["source"].get("source_split") == "test":
        raise ValueError("test source cannot be used for training or checkpoint selection")
    if selecting and case["source"].get("concise_test_overlap") is True:
        raise ValueError("concise-test source cannot be used for training or checkpoint selection")
    if selecting and case["source"].get("dataset") == "Talk2Car-RegSeg":
        if case["source"].get("source_split") != "train":
            raise ValueError("RegSeg validation/test sources are reserved from checkpoint selection")
        if case["source"].get("scene_group_verified") is not True:
            raise ValueError("RegSeg source-scene grouping must be verified before model development")
    result = deepcopy(case)
    result["request"] = validate_request(case["request"])
    frame = result["request"]["scene"]["frame"]
    if result["request"]["request_id"] != result["case_id"]:
        raise ValueError("request_id must match case_id")
    result["target"] = validate_target(case["target"], case["supervised_fields"], frame)
    if selecting and result["source"].get("dataset") == "Talk2Car-RegSeg":
        _regseg_group_guard(result)
    if (result["source"].get("dataset") == "Talk2Car-RegSeg"
            and result["review"]["status"] == "public_annotation"
            and set(result["supervised_fields"]) != DESTINATION_FIELDS):
        raise ValueError("public RegSeg masks supervise destination fields only, not status or anchors")
    _human_review_guard(result)
    base = Path(base_dir).resolve() if base_dir is not None else Path.cwd()
    image_path = _resolve(frame["image_path"], base)
    frame["image_path"] = str(image_path)
    if not image_path.is_file():
        raise ValueError("case image is unavailable")
    source = result["source"]
    source["image_sha256"] = _verify_hash(image_path, source.get("image_sha256"), "image")
    if verify_images:
        from PIL import Image
        with Image.open(image_path) as image:
            if image.size != (frame["width"], frame["height"]):
                raise ValueError("image dimensions differ from the scene")
            image.load()
    if source.get("mask_path") is not None:
        mask_path = _resolve(source["mask_path"], base)
        if not mask_path.is_file():
            raise ValueError("case mask is unavailable")
        source["mask_path"] = str(mask_path)
        source["mask_sha256"] = _verify_hash(mask_path, source.get("mask_sha256"), "mask")
        if verify_images:
            from PIL import Image
            with Image.open(mask_path) as mask_image:
                mask = mask_image.convert("L")
                bounds = mask.getbbox()
                if bounds is None:
                    raise ValueError("destination mask is empty")
                point = result["target"].get("destination_point_xy")
                if point is not None:
                    mask_x = math.floor(point[0] * mask.width / frame["width"])
                    mask_y = math.floor(point[1] * mask.height / frame["height"])
                    if mask.getpixel((mask_x, mask_y)) == 0:
                        raise ValueError("reference point is outside the source destination mask")
                box = result["target"].get("destination_box_xyxy")
                if box is not None:
                    expected = [bounds[0] * frame["width"] / mask.width,
                                bounds[1] * frame["height"] / mask.height,
                                bounds[2] * frame["width"] / mask.width,
                                bounds[3] * frame["height"] / mask.height]
                    if any(abs(a - b) > 1e-6 for a, b in zip(box, expected)):
                        raise ValueError("destination box does not match its source mask extent")
    return result


def load_cases(path, *, allowed_partitions=("train",), verify_images=True, payload=None):
    """Read one explicit split; refuse excluded rows rather than silently filtering."""
    path = Path(path)
    rows = []
    seen = set()
    if payload is None:
        if path.stat().st_size > 128 * 1024 * 1024:
            raise ValueError("case file exceeds the snapshot byte budget")
        payload = path.read_bytes()
    if not isinstance(payload, bytes) or len(payload) > 128 * 1024 * 1024:
        raise ValueError("case file exceeds the snapshot byte budget")
    for line_number, line in enumerate(payload.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        if len(line) > 2_000_000 or len(rows) >= 50000:
            raise ValueError("case file exceeds bounded input limits")
        try:
            row = validate_case(json.loads(line, object_pairs_hook=_unique_object), base_dir=path.parent,
                                allowed_partitions=allowed_partitions, verify_images=verify_images)
        except (ValueError, OSError) as error:
            raise ValueError(f"{path.name}:{line_number}: {error}") from error
        if row["case_id"] in seen:
            raise ValueError(f"duplicate case_id: {row['case_id']}")
        seen.add(row["case_id"])
        rows.append(row)
    if not rows:
        raise ValueError("case file has no usable examples")
    return rows


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def validate_disjoint_groups(train_cases, development_cases):
    """Prevent scene/identity/image duplication across training and selection data."""
    for label, field in (("case identity", "case_id"), ("scene group", "scene_group")):
        overlap = {row[field] for row in train_cases} & {row[field] for row in development_cases}
        if overlap:
            raise ValueError(f"training/development {label} overlap: {sorted(overlap)[:5]}")
    overlap = {row["source"]["image_sha256"] for row in train_cases} & {
        row["source"]["image_sha256"] for row in development_cases
    }
    if overlap:
        raise ValueError("training/development contains identical image bytes")
