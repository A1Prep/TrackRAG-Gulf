"""Task 4 relevance review packs and split-safe pointwise dataset exports.

No model or existing answer key supplies labels here. Preparation snapshots the
candidate pool and image bytes; every judgment starts pending. Export accepts
explicit human relevant/irrelevant judgments for one recording partition, keeping
labels and reviewer information outside the model-input fields.
"""

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import re
import tempfile

from PIL import Image

from .contracts import TrackObservation
from .evidence_time import requested_frame_ids
from .questions import validate_question
from .retrieval import RetrievalConfig, _track_windows
from .retrieval_candidates import (
    SCHEMA_VERSION, CandidateBatch, CandidateImage, CandidatePolicy, EvidenceCandidate,
    ObservedFeatures, TrackHistory, _canonical, _identifier,
)
from .retrieval_review_html import render_review


PACK_VERSION = "trackrag.relevance-review.v1"
LABEL_FIELDS = {
    "pack_sha256", "case_id", "candidate_id", "judgment", "reviewer", "reviewer_type",
    "reason", "note", "checked_frame_ids", "reviewed_at",
}
REASONS = {
    "relevant": {"supports_question", "other"},
    "irrelevant": {"wrong_vehicle", "wrong_time", "unrelated_scene", "insufficient_evidence", "other"},
    "uncertain": {"ambiguous", "insufficient_evidence", "other"},
}


def _hash(value):
    return sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_hash(path):
    with Path(path).open("rb") as stream:
        digest = sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


def _loads(text):
    return json.loads(text, object_pairs_hook=_object, parse_constant=_reject_constant)


def _read(path):
    return _loads(Path(path).read_text(encoding="utf-8-sig"))


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _fields(value, names, description):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(f"{description} must contain exactly its documented fields")


def _integer(value, description):
    if type(value) is not int or value < 0:
        raise ValueError(f"{description} must be a nonnegative integer")
    return value


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _box(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 4 or not all(_number(x) for x in value)
            or not (0 <= value[0] < value[2] and 0 <= value[1] < value[3])):
        raise ValueError("Invalid original-pixel observation box")


def _splits(manifest):
    _fields(manifest, ("train", "dev", "test"), "Split manifest")
    assigned = {}
    for split, recordings in manifest.items():
        if not isinstance(recordings, list):
            raise ValueError("Split recordings must be lists")
        for recording in recordings:
            if not isinstance(recording, str) or not recording.strip() or recording in assigned:
                raise ValueError("Recording splits must be nonempty IDs without overlap or duplicates")
            assigned[recording] = split
    return assigned


def _features(features, rows, max_gap):
    _fields(features, ObservedFeatures.__required_keys__, "Observed features")
    ids = [row["frame_id"] for row in rows]
    gaps = [[a + 1, b - 1] for a, b in zip(ids, ids[1:]) if b > a + 1]
    breaks = [gap for gap in gaps if gap[1] - gap[0] + 1 > max_gap]
    if (features["observed_frame_ids"] != ids or type(features["observation_count"]) is not int
            or features["observation_count"] != len(rows)
            or features["class_counts"] != dict(Counter(row["class_name"] for row in rows))
            or any(type(count) is not int for count in features["class_counts"].values())
            or list(features["first_box_xyxy"]) != list(rows[0]["bbox_xyxy"])
            or list(features["last_box_xyxy"]) != list(rows[-1]["bbox_xyxy"])
            or [list(gap) for gap in features["missing_observation_ranges"]] != gaps
            or [list(gap) for gap in features["continuity_breaks"]] != breaks):
        raise ValueError("Candidate features disagree with the supplied observations")
    for key in ("first_center_normalized", "last_center_normalized", "center_range_normalized"):
        value = features[key]
        if not isinstance(value, (list, tuple)) or len(value) != 2 or not all(_number(x) and 0 <= x <= 1 for x in value):
            raise ValueError("Invalid normalized candidate features")
    delta = features["center_displacement_normalized"]
    if len(rows) == 1:
        if delta is not None:
            raise ValueError("One observation cannot establish displacement")
    elif (not isinstance(delta, (list, tuple)) or len(delta) != 2
          or not all(_number(x) and -1 <= x <= 1 for x in delta)):
        raise ValueError("Invalid normalized displacement")


def _metadata_hash(batch):
    value = deepcopy(batch)
    del value["metadata_sha256"]
    for candidate in value["candidates"]:
        for image in candidate["images"]:
            del image["image_path"]
    return _hash(value)


def _verify_geometry(features, rows, images):
    """Verify centers from sampled pixels; do not invent unsampled dimensions."""
    lookup = {image["frame_id"]: image for image in images}
    centers = []
    for row in rows:
        image = lookup.get(row["frame_id"])
        if image is None:
            continue
        left, top, right, bottom = row["bbox_xyxy"]
        centers.append(((left / image["width"] + right / image["width"]) / 2,
                        (top / image["height"] + bottom / image["height"]) / 2))

    def agrees(actual, expected):
        return all(math.isclose(a, b, rel_tol=0, abs_tol=1e-12) for a, b in zip(actual, expected))

    if (not agrees(features["first_center_normalized"], centers[0])
            or not agrees(features["last_center_normalized"], centers[-1])):
        raise ValueError("Normalized endpoint geometry disagrees with original boxes")
    if len(rows) > 1 and not agrees(features["center_displacement_normalized"],
                                    [centers[-1][axis] - centers[0][axis] for axis in (0, 1)]):
        raise ValueError("Displacement disagrees with the observed endpoint centers")
    sampled_range = [max(point[axis] for point in centers) - min(point[axis] for point in centers) for axis in (0, 1)]
    claimed_range = features["center_range_normalized"]
    if (any(claimed + 1e-12 < sampled for claimed, sampled in zip(claimed_range, sampled_range))
            or (len(centers) == len(rows) and not agrees(claimed_range, sampled_range))):
        raise ValueError("Center range contradicts sampled observation geometry")


def _validate_batch(batch):
    """Check membership, complete pool shape and time bounds, not factual relevance."""
    _fields(batch, CandidateBatch.__required_keys__, "Candidate batch")
    if batch["schema_version"] != SCHEMA_VERSION or batch["source"] not in ("annotated", "predicted"):
        raise ValueError("Unsupported candidate schema or source")
    question = validate_question(batch["question"])
    recording, source, cutoff = question["recording_id"], batch["source"], question["query_frame"]
    policy = batch["policy"]
    _fields(policy, CandidatePolicy.__required_keys__, "Candidate policy")
    config = RetrievalConfig(window_frames=policy["window_frames"], stride_frames=policy["stride_frames"],
                             max_gap_frames=policy["max_gap_frames"])
    if (policy["scenes"] != "query, named past frames, earliest permitted observation"
            or policy["images"] != "first, middle floor((n-1)/2), last, explicitly requested observed frames"):
        raise ValueError("Unsupported candidate selection policy")
    histories, expected = {}, set()
    for history in batch["histories"]:
        _fields(history, TrackHistory.__required_keys__, "Track history")
        track = _integer(history["track_id"], "track_id")
        hid = _identifier("H-", recording, source, track)
        if (history["history_id"] != hid or hid in histories
                or history["recording_id"] != recording or history["source"] != source):
            raise ValueError("History identity/source does not match the public question")
        rows = history["observations"]
        if not isinstance(rows, list) or not rows:
            raise ValueError("Track histories require observations")
        ids = []
        for row in rows:
            _fields(row, TrackObservation.__required_keys__, "Track observation")
            frame = _integer(row["frame_id"], "observation frame")
            if (frame > cutoff or row["recording_id"] != recording or row["source"] != source
                    or type(row["track_id"]) is not int or row["track_id"] != track):
                raise ValueError("Observation has a foreign identity/source or future frame")
            if not isinstance(row["class_name"], str) or not row["class_name"].strip():
                raise ValueError("Observation class must be nonempty")
            _box(row["bbox_xyxy"])
            ids.append(frame)
        if ids != sorted(set(ids)):
            raise ValueError("History frames must be unique and ordered")
        _features(history["features"], rows, config.max_gap_frames)
        histories[hid] = history
        full = (ids[0], ids[-1])
        expected.add(_identifier("C-", recording, source, "track_history", track, *full))
        for window in _track_windows(tuple(rows), config):
            if (window.start_frame, window.end_frame) != full:
                expected.add(_identifier("C-", recording, source, "track_window", track, window.start_frame, window.end_frame))
    requested = set(requested_frame_ids(question["question"]))
    allowed = {frame for frame in requested if frame <= cutoff}
    scenes = allowed | {cutoff}
    if histories:
        scenes.add(min(history["observations"][0]["frame_id"] for history in histories.values()))
    unavailable = batch["unavailable_scene_frames"]
    if (not isinstance(unavailable, list) or any(type(frame) is not int for frame in unavailable)
            or unavailable != sorted(set(unavailable)) or not set(unavailable) <= scenes
            or batch["excluded_future_frame_requests"] != sorted(requested - allowed)):
        raise ValueError("Invalid missing/future scene-frame accounting")
    expected.update(_identifier("C-", recording, source, "scene", None, frame, frame) for frame in scenes - set(unavailable))
    seen = set()
    for candidate in batch["candidates"]:
        _fields(candidate, EvidenceCandidate.__required_keys__, "Evidence candidate")
        start = _integer(candidate["start_frame"], "start_frame")
        end = _integer(candidate["end_frame"], "end_frame")
        kind, track = candidate["kind"], candidate["track_id"]
        if (not start <= end <= cutoff or candidate["recording_id"] != recording
                or candidate["source"] != source or kind not in ("track_history", "track_window", "scene")):
            raise ValueError("Candidate has invalid identity/source or future bounds")
        cid = _identifier("C-", recording, source, kind, track, start, end)
        if candidate["candidate_id"] != cid or cid in seen:
            raise ValueError("Invalid or duplicate candidate ID")
        seen.add(cid)
        if kind == "scene":
            if track is not None or start != end or candidate["history_id"] is not None or candidate["features"] is not None:
                raise ValueError("Scene candidates require exact frames and null track/box context")
            selected, lookup = {start}, {}
        else:
            _integer(track, "candidate track")
            history = histories.get(candidate["history_id"])
            if history is None or history["track_id"] != track:
                raise ValueError("Candidate history membership mismatch")
            rows = [row for row in history["observations"] if start <= row["frame_id"] <= end]
            if not rows or (rows[0]["frame_id"], rows[-1]["frame_id"]) != (start, end):
                raise ValueError("Candidate endpoints are not observed")
            _features(candidate["features"], rows, config.max_gap_frames)
            lookup = {row["frame_id"]: row for row in rows}
            selected = {rows[i]["frame_id"] for i in (0, (len(rows)-1)//2, len(rows)-1)} | (allowed & lookup.keys())
        image_ids = []
        for image in candidate["images"]:
            _fields(image, CandidateImage.__required_keys__, "Candidate image")
            frame = _integer(image["frame_id"], "image frame")
            if image["recording_id"] != recording or not start <= frame <= end <= cutoff:
                raise ValueError("Image has a foreign recording or future/outside-interval frame")
            if any(type(image[key]) is not int or image[key] <= 0 for key in ("width", "height")):
                raise ValueError("Image dimensions must be positive integers")
            if not _number(image["timestamp_s"]) or image["timestamp_s"] < 0:
                raise ValueError("Image timestamp must be finite and nonnegative")
            if not isinstance(image["image_path"], str) or not image["image_path"].strip():
                raise ValueError("Image path must be nonempty")
            if kind == "scene":
                if image["bbox_xyxy"] is not None:
                    raise ValueError("Scene cannot have a target box")
            else:
                _box(image["bbox_xyxy"])
                if (frame not in lookup or list(image["bbox_xyxy"]) != list(lookup[frame]["bbox_xyxy"])
                        or image["bbox_xyxy"][2] > image["width"] or image["bbox_xyxy"][3] > image["height"]):
                    raise ValueError("Image box does not match its observed track/frame")
            image_ids.append(frame)
        if image_ids != sorted(selected):
            raise ValueError("Image references do not match the fixed candidate sampling policy")
        if kind != "scene":
            _verify_geometry(candidate["features"], rows, candidate["images"])
            if kind == "track_history":
                _verify_geometry(history["features"], history["observations"], candidate["images"])
    if seen != expected:
        raise ValueError("Candidate pool is incomplete or contains unsupported intervals")
    if batch["metadata_sha256"] != _metadata_hash(batch):
        raise ValueError("Candidate metadata hash does not match the packet")
    return batch


def _asset_path(root, name):
    if not isinstance(name, str) or not re.fullmatch(r"images/[0-9a-f]{64}\.(png|jpg)", name):
        raise ValueError("Invalid snapshot image path")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Image path escapes the review pack")
    return path


def _pack_hash(pack):
    return _hash({key: value for key, value in pack.items() if key != "pack_sha256"})


def _pending(pack_hash, case_id, candidate_id):
    return {"pack_sha256": pack_hash, "case_id": case_id, "candidate_id": candidate_id,
            "judgment": "pending", "reviewer": "", "reviewer_type": None, "reason": None,
            "note": "", "checked_frame_ids": [], "reviewed_at": None}


def prepare_review(candidate_paths, split_manifest_path, output_dir):
    """Snapshot fixed candidate pools/images and create a completely ungraded pack."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("Review output already exists")
    assigned_manifest = _read(split_manifest_path)
    assigned = _splits(assigned_manifest)
    batches = [_validate_batch(_read(path)) for path in candidate_paths]
    if not batches or not any(batch["candidates"] for batch in batches):
        raise ValueError("At least one candidate is required for review")
    if len({_hash(batch["policy"]) for batch in batches}) != 1:
        raise ValueError("A review pack must use one fixed candidate policy")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".task4-review-", dir=output.parent) as temporary:
        staging = Path(temporary) / "pack"
        (staging / "images").mkdir(parents=True)
        (staging / "batches").mkdir()
        cases, packets, assets, cached, questions = [], {}, {}, {}, {}
        frame_contents = {}
        for batch in batches:
            q = batch["question"]
            if q["recording_id"] not in assigned:
                raise ValueError("Recording has no declared split")
            if q["question_id"] in questions and questions[q["question_id"]] != q:
                raise ValueError("A question ID cannot identify different questions")
            questions[q["question_id"]] = q
            case_id = _hash([q["recording_id"], batch["source"], q["question_id"]])
            if case_id in packets:
                raise ValueError("Duplicate question/source case")
            copied = deepcopy(batch)
            for candidate in copied["candidates"]:
                for image in candidate["images"]:
                    original = Path(image["image_path"]).resolve()
                    if original not in cached:
                        payload = original.read_bytes()
                        try:
                            with Image.open(io.BytesIO(payload)) as opened:
                                size, image_format = opened.size, opened.format
                                opened.verify()
                        except (OSError, SyntaxError) as error:
                            raise ValueError(f"Cannot decode candidate image: {original}") from error
                        if image_format not in ("PNG", "JPEG"):
                            raise ValueError("Review images must be PNG or JPEG")
                        digest = sha256(payload).hexdigest()
                        relative = "images/" + digest + (".png" if image_format == "PNG" else ".jpg")
                        if relative not in assets:
                            (staging / relative).write_bytes(payload)
                            assets[relative] = {"path": relative, "sha256": digest}
                        cached[original] = (relative, size)
                    relative, size = cached[original]
                    if size != (image["width"], image["height"]):
                        raise ValueError("Original image dimensions disagree with candidate metadata")
                    frame_key = (q["recording_id"], image["frame_id"])
                    if frame_key in frame_contents and frame_contents[frame_key] != relative:
                        raise ValueError("The same recording/frame has inconsistent image contents")
                    frame_contents[frame_key] = relative
                    image["image_path"] = relative
            batch_file = f"batches/{case_id}.json"
            _write(staging / batch_file, copied)
            packets[case_id] = copied
            cases.append({"case_id": case_id, "question_id": q["question_id"], "recording_id": q["recording_id"],
                          "source": batch["source"], "split": assigned[q["recording_id"]],
                          "metadata_sha256": batch["metadata_sha256"], "batch_file": batch_file,
                          "batch_sha256": _file_hash(staging / batch_file)})
        pack = {"schema_version": PACK_VERSION, "split_manifest": assigned_manifest,
                "cases": cases, "assets": sorted(assets.values(), key=lambda item: item["path"])}
        pack["pack_sha256"] = _pack_hash(pack)
        labels = [_pending(pack["pack_sha256"], case["case_id"], candidate["candidate_id"])
                  for case in cases for candidate in packets[case["case_id"]]["candidates"]]
        _write(staging / "pack.json", pack)
        (staging / "labels.jsonl").write_text("".join(_canonical(row) + "\n" for row in labels), encoding="utf-8")
        (staging / "review.html").write_text(render_review(pack, packets, labels), encoding="utf-8")
        if output.exists():
            raise FileExistsError("Review output was created during preparation")
        staging.rename(output)
    return pack


def _load_pack(directory):
    root = Path(directory).resolve()
    pack = _read(root / "pack.json")
    _fields(pack, ("schema_version", "split_manifest", "cases", "assets", "pack_sha256"), "Review pack")
    if pack["schema_version"] != PACK_VERSION or pack["pack_sha256"] != _pack_hash(pack):
        raise ValueError("Review pack hash/schema mismatch")
    assigned = _splits(pack["split_manifest"])
    assets = {}
    for asset in pack["assets"]:
        _fields(asset, ("path", "sha256"), "Image asset")
        path = _asset_path(root, asset["path"])
        if asset["path"] in assets or _file_hash(path) != asset["sha256"]:
            raise ValueError("Review image hash mismatch or duplicate asset")
        assets[asset["path"]] = asset
    packets, lookup, referenced = {}, {}, set()
    for case in pack["cases"]:
        cid = case["case_id"]
        if (not isinstance(cid, str) or not re.fullmatch(r"[0-9a-f]{64}", cid)
                or cid in packets or case["batch_file"] != f"batches/{cid}.json"):
            raise ValueError("Invalid/duplicate case or batch path")
        path = (root / case["batch_file"]).resolve()
        if not path.is_relative_to(root) or _file_hash(path) != case["batch_sha256"]:
            raise ValueError("Candidate batch file hash mismatch")
        batch = _validate_batch(_read(path))
        q = batch["question"]
        if (cid != _hash([q["recording_id"], batch["source"], q["question_id"]])
                or case["question_id"] != q["question_id"] or case["recording_id"] != q["recording_id"]
                or case["source"] != batch["source"] or case["split"] != assigned.get(q["recording_id"])
                or case["metadata_sha256"] != batch["metadata_sha256"]):
            raise ValueError("Case identity, split or metadata binding mismatch")
        packets[cid] = batch
        for candidate in batch["candidates"]:
            lookup[(cid, candidate["candidate_id"])] = (case, batch, candidate)
            for image in candidate["images"]:
                if image["image_path"] not in assets:
                    raise ValueError("Candidate references an unbound image")
                referenced.add(image["image_path"])
    if referenced != set(assets):
        raise ValueError("Review image inventory differs from candidate references")
    return pack, lookup


def _validate_labels(path, pack, lookup):
    rows = [_loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    labels = {}
    for row in rows:
        _fields(row, LABEL_FIELDS, "Review judgment")
        key = (row["case_id"], row["candidate_id"])
        if row["pack_sha256"] != pack["pack_sha256"] or key not in lookup or key in labels:
            raise ValueError("Unknown, duplicate or stale review judgment")
        if row["judgment"] not in ("pending", "relevant", "irrelevant", "uncertain"):
            raise ValueError("Invalid relevance judgment")
        if not isinstance(row["reviewer"], str) or not isinstance(row["note"], str):
            raise ValueError("Reviewer and rationale must be strings")
        if row["reviewer_type"] not in (None, "human", "ai_assisted"):
            raise ValueError("Unknown reviewer type")
        checked = row["checked_frame_ids"]
        available = sorted(image["frame_id"] for image in lookup[key][2]["images"])
        if (not isinstance(checked, list) or any(type(frame) is not int for frame in checked)
                or len(set(checked)) != len(checked) or not set(checked) <= set(available)):
            raise ValueError("Invalid inspected frame IDs")
        if row["judgment"] != "pending":
            if (row["reviewer_type"] is None or not row["reviewer"].strip() or not row["note"].strip()
                    or row["reason"] not in REASONS[row["judgment"]] or sorted(checked) != available):
                raise ValueError("Completed review requires reviewer, rationale, reason and all inspected frames")
            try:
                stamp = datetime.fromisoformat(row["reviewed_at"])
            except (TypeError, ValueError) as error:
                raise ValueError("Completed review requires a valid reviewed_at timestamp") from error
            if stamp.tzinfo is None:
                raise ValueError("Review timestamp must include a timezone")
        labels[key] = row
    if set(labels) != set(lookup):
        raise ValueError("Keep every candidate judgment, including pending rows; review coverage is incomplete")
    return labels


def export_reviewed(review_dir, labels_path, output_dir, *, split):
    """Export human-labeled pointwise examples for exactly one declared split.

    Pending, uncertain and AI-assisted rows never become negative labels. Partial
    review is allowed but quantified, and zero eligible rows is an error. A claim
    of human review is recorded provenance, not authenticated human identity.
    """
    if split not in ("train", "dev", "test"):
        raise ValueError("Explicit split must be train, dev or test")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("Dataset output already exists")
    pack, lookup = _load_pack(review_dir)
    labels = _validate_labels(labels_path, pack, lookup)
    examples, counts, selected_counts = [], Counter(), Counter()
    per_case = {case["case_id"]: {
        "case_id": case["case_id"], "question_id": case["question_id"], "recording_id": case["recording_id"],
        "source": case["source"], "split": case["split"], "candidate_count": 0,
        "pending": 0, "uncertain": 0, "human_relevant": 0, "human_irrelevant": 0, "ai_assisted": 0,
    } for case in pack["cases"]}
    for key, (case, batch, candidate) in lookup.items():
        row = labels[key]
        counts[row["judgment"]] += 1
        stats = per_case[case["case_id"]]
        stats["candidate_count"] += 1
        if row["judgment"] in ("pending", "uncertain"):
            stats[row["judgment"]] += 1
        if row["judgment"] != "pending" and row["reviewer_type"] == "ai_assisted":
            stats["ai_assisted"] += 1
        if row["judgment"] in ("relevant", "irrelevant") and row["reviewer_type"] == "human":
            stats["human_" + row["judgment"]] += 1
        if case["split"] != split:
            continue
        selected_counts[row["judgment"]] += 1
        if row["judgment"] not in ("relevant", "irrelevant") or row["reviewer_type"] != "human":
            continue
        history = next((h for h in batch["histories"] if h["history_id"] == candidate["history_id"]), None)
        examples.append({
            "model_input": {"question": deepcopy(batch["question"]), "source": batch["source"],
                            "candidate": deepcopy(candidate), "history": deepcopy(history)},
            "relevance": int(row["judgment"] == "relevant"),
            "provenance": {"pack_sha256": pack["pack_sha256"], "case_id": case["case_id"],
                           "candidate_id": candidate["candidate_id"], "split": split,
                           "metadata_sha256": case["metadata_sha256"],
                           "review": deepcopy(row)},
        })
    if not examples:
        raise ValueError(f"No reviewed human examples are available for split {split}; pending/uncertain/AI rows are excluded")
    for stats in per_case.values():
        stats["empty_pool"] = stats["candidate_count"] == 0
        stats["fully_human_labeled"] = not stats["empty_pool"] and stats["human_relevant"] + stats["human_irrelevant"] == stats["candidate_count"]
        stats["no_positive_reviewed_yet"] = stats["human_relevant"] == 0
        stats["fully_reviewed_without_positive"] = stats["fully_human_labeled"] and stats["human_relevant"] == 0
    wanted_assets = {image["image_path"] for example in examples for image in example["model_input"]["candidate"]["images"]}
    assets = [asset for asset in pack["assets"] if asset["path"] in wanted_assets]
    summary = {
        "schema_version": "trackrag.relevance-dataset.v1", "split": split, "pack_sha256": pack["pack_sha256"],
        "labels_sha256": _file_hash(labels_path), "exported_examples": len(examples),
        "positive_examples": sum(example["relevance"] for example in examples),
        "negative_examples": sum(not example["relevance"] for example in examples),
        "candidate_count": len(lookup), "judgment_counts": dict(counts),
        "selected_split_judgment_counts": dict(selected_counts), "questions": list(per_case.values()),
        "split_manifest": pack["split_manifest"], "assets": assets,
        "limitations": ["Human reviewer identity is declared, not authenticated.",
                        "Unreviewed, uncertain and AI-assisted rows are excluded, not assigned negative labels.",
                        "Fully reviewed without a positive does not prove a candidate-pool miss; the question may be unanswerable or ambiguous.",
                        "A split declaration does not establish an untouched held-out evaluation set.",
                        "Only model_input belongs in a model prompt; relevance and provenance are training/evaluation metadata."],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".task4-dataset-", dir=output.parent) as temporary:
        staging = Path(temporary) / "dataset"
        (staging / "images").mkdir(parents=True)
        for asset in assets:
            payload = _asset_path(Path(review_dir), asset["path"]).read_bytes()
            if sha256(payload).hexdigest() != asset["sha256"]:
                raise ValueError("Review image changed during export")
            (staging / asset["path"]).write_bytes(payload)
        (staging / "examples.jsonl").write_text("".join(_canonical(row) + "\n" for row in examples), encoding="utf-8")
        summary["examples_sha256"] = _file_hash(staging / "examples.jsonl")
        _write(staging / "summary.json", summary)
        if output.exists():
            raise FileExistsError("Dataset output was created during export")
        staging.rename(output)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Snapshot candidates/images and create pending relevance reviews")
    prepare.add_argument("--candidates", nargs="+", type=Path, required=True)
    prepare.add_argument("--split-manifest", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    export = commands.add_parser("export", help="Export explicitly reviewed human labels for one split")
    export.add_argument("--review", type=Path, required=True)
    export.add_argument("--labels", type=Path, required=True)
    export.add_argument("--split", choices=("train", "dev", "test"), required=True)
    export.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            pack = prepare_review(args.candidates, args.split_manifest, args.out)
            print(f"Prepared {len(pack['cases'])} question/source cases at {args.out / 'review.html'}; all judgments are pending.")
        else:
            summary = export_reviewed(args.review, args.labels, args.out, split=args.split)
            print(f"Exported {summary['exported_examples']} reviewed examples for {args.split} to {args.out}")
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
