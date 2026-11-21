"""Validated Task 4 training bundles, with synthetic and reviewed data kept distinct."""

from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from .retrieval import build_index
from .retrieval_candidates import build_candidate_batch
from .retrieval_dataset import _load_pack, _loads, _splits, _validate_batch, _validate_labels
from .retrieval_qwen import INPUT_VERSION, text_documents


SCHEMA_VERSION = "trackrag.qwen-training-data.v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return sha256(canonical(value).encode("utf-8")).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _seal(bundle):
    value = deepcopy(bundle)
    value.pop("dataset_sha256", None)
    bundle["dataset_sha256"] = digest(value)
    return bundle


def validate_dataset(bundle):
    """Return permitted question/document/label samples, never label-bearing prompts.

    Full packets reuse the inference validator and formatter. Only train/dev
    recording partitions are accepted. Provenance is a recorded assertion, not
    authentication that a human really inspected the images.
    """
    required = {"schema_version", "input_version", "kind", "split_manifest", "batches",
                "examples", "provenance", "dataset_sha256"}
    if not isinstance(bundle, dict) or set(bundle) != required:
        raise ValueError("Training bundle must contain exactly the documented fields")
    if bundle["schema_version"] != SCHEMA_VERSION or bundle["input_version"] != INPUT_VERSION:
        raise ValueError("Unsupported training data/input version")
    if bundle["kind"] not in ("synthetic_smoke", "human_reviewed"):
        raise ValueError("Training data must explicitly declare synthetic_smoke or human_reviewed")
    copy = deepcopy(bundle)
    claimed = copy.pop("dataset_sha256")
    if claimed != digest(copy):
        raise ValueError("Training bundle hash mismatch")
    assigned = _splits(bundle["split_manifest"])
    if not isinstance(bundle["batches"], list) or not isinstance(bundle["examples"], list):
        raise ValueError("Training batches and examples must be lists")
    packets, documents, questions = {}, {}, set()
    for batch in bundle["batches"]:
        _validate_batch(batch)
        identity = batch["metadata_sha256"]
        query = batch["question"]
        question_key = (query["recording_id"], batch["source"], query["question_id"])
        if identity in packets or question_key in questions:
            raise ValueError("Duplicate or conflicting training question packet")
        if assigned.get(query["recording_id"]) not in ("train", "dev"):
            raise ValueError("Test or unassigned recordings must not enter training bundles")
        questions.add(question_key)
        packets[identity] = batch
        documents[identity] = dict(zip((candidate["candidate_id"] for candidate in batch["candidates"]),
                                       text_documents(batch)))
    samples, seen, used_packets = [], set(), set()
    for row in bundle["examples"]:
        if not isinstance(row, dict) or set(row) != {"metadata_sha256", "candidate_id", "split", "relevance", "label_provenance"}:
            raise ValueError("Invalid training example fields")
        identity, candidate_id = row["metadata_sha256"], row["candidate_id"]
        key = (identity, candidate_id)
        if identity not in packets or candidate_id not in documents[identity] or key in seen:
            raise ValueError("Unknown or duplicate training candidate binding")
        batch = packets[identity]
        if row["split"] != assigned[batch["question"]["recording_id"]]:
            raise ValueError("Training example disagrees with its recording partition")
        if type(row["relevance"]) is not int or row["relevance"] not in (0, 1):
            raise ValueError("Relevance must be the integer 0 or 1")
        label = row["label_provenance"]
        if not isinstance(label, dict):
            raise ValueError("Label provenance must be explicit")
        if bundle["kind"] == "synthetic_smoke":
            if label.get("type") != "synthetic" or not batch["question"]["recording_id"].startswith("synthetic-"):
                raise ValueError("Synthetic smoke labels must identify generated recordings")
        else:
            review = label.get("review", {})
            judgment = "relevant" if row["relevance"] else "irrelevant"
            if (label.get("type") != "human" or review.get("reviewer_type") != "human"
                    or review.get("judgment") != judgment or not review.get("reviewer", "").strip()
                    or review.get("candidate_id") != candidate_id
                    or review.get("pack_sha256") != bundle["provenance"].get("pack_sha256")):
                raise ValueError("Unreviewed, uncertain or AI-assisted labels are not reviewed training examples")
        seen.add(key)
        used_packets.add(identity)
        samples.append({"sample_id": digest(key), "question": deepcopy(batch["question"]),
                        "document": documents[identity][candidate_id], "relevance": row["relevance"],
                        "split": row["split"], "candidate_id": candidate_id})
    if used_packets != set(packets):
        raise ValueError("Training bundle contains unused packets")
    train = [sample for sample in samples if sample["split"] == "train"]
    dev = [sample for sample in samples if sample["split"] == "dev"]
    if {sample["relevance"] for sample in train} != {0, 1} or not dev:
        raise ValueError("Supply positive and negative train examples plus separate dev examples")
    return samples


def load_dataset(path):
    bundle = _loads(Path(path).read_text(encoding="utf-8-sig"))
    validate_dataset(bundle)
    return bundle


def prepare_reviewed(review_dir, labels_path, output_path, *, tracked_only=False):
    """Use the existing hash/image/label checks; preserve its human-only policy."""
    output = Path(output_path)
    if output.exists():
        raise FileExistsError("Choose a new training dataset path")
    pack, lookup = _load_pack(review_dir)
    labels = _validate_labels(labels_path, pack, lookup)
    packets, examples, counts = {}, [], Counter()
    for key, (case, batch, candidate) in lookup.items():
        label = labels[key]
        if tracked_only and candidate["kind"] == "scene":
            counts["excluded_scene"] += 1
            counts["excluded"] += 1
            continue
        if (case["split"] not in ("train", "dev") or label["reviewer_type"] != "human"
                or label["judgment"] not in ("relevant", "irrelevant")):
            counts["excluded"] += 1
            continue
        identity = batch["metadata_sha256"]
        packets[identity] = batch
        examples.append({"metadata_sha256": identity, "candidate_id": candidate["candidate_id"],
                         "split": case["split"], "relevance": int(label["judgment"] == "relevant"),
                         "label_provenance": {"type": "human", "review": deepcopy(label)}})
        counts[case["split"]] += 1
    bundle = _seal({"schema_version": SCHEMA_VERSION, "input_version": INPUT_VERSION,
                    "kind": "human_reviewed", "split_manifest": deepcopy(pack["split_manifest"]),
                    "batches": list(packets.values()), "examples": examples,
                    "provenance": {"pack_sha256": pack["pack_sha256"],
                                   "labels_file_sha256": sha256(Path(labels_path).read_bytes()).hexdigest(),
                                   "counts": dict(counts), "tracked_only": tracked_only,
                                   "human_identity_is_declared": True}})
    validate_dataset(bundle)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects an existing dataset even if another writer races.
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(bundle, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return bundle


def synthetic_dataset():
    """Two generated recordings, four class-only labels: a machinery check only."""
    class GeneratedRecording:
        def __init__(self, identity):
            self.recording_id = identity

        def get_frame(self, frame_id):
            if frame_id != 1:
                raise KeyError(frame_id)
            return {"recording_id": self.recording_id, "frame_id": 1, "timestamp_s": 0.0,
                    "width": 100, "height": 80, "image_path": "synthetic-not-read.jpg"}

        def iter_frames(self):
            raise AssertionError("No footage is read for synthetic training")

    batches, examples = [], []
    for split, target in (("train", "bus"), ("dev", "car")):
        identity = "synthetic-" + split
        recording = GeneratedRecording(identity)
        rows = [{"recording_id": identity, "source": "annotated", "frame_id": 1,
                 "track_id": track, "class_name": label, "bbox_xyxy": box}
                for track, label, box in ((7, "Bus", (10, 20, 30, 40)), (42, "Car", (60, 20, 80, 40)))]
        query = {"question_id": "synthetic-" + split, "recording_id": identity,
                 "query_frame": 1, "question": f"Which {target} is visible?"}
        batch = build_candidate_batch(query, build_index(recording, rows), source="annotated")
        batches.append(batch)
        for candidate in batch["candidates"]:
            if candidate["track_id"] is None:
                continue
            examples.append({"metadata_sha256": batch["metadata_sha256"], "candidate_id": candidate["candidate_id"],
                             "split": split, "relevance": int(candidate["track_id"] == (7 if target == "bus" else 42)),
                             "label_provenance": {"type": "synthetic", "rule": "Known class in generated observations; not human review"}})
    bundle = _seal({"schema_version": SCHEMA_VERSION, "input_version": INPUT_VERSION, "kind": "synthetic_smoke",
                    "split_manifest": {"train": ["synthetic-train"], "dev": ["synthetic-dev"], "test": []},
                    "batches": batches, "examples": examples,
                    "provenance": {"purpose": "Validate optimizer updates and save/reload; no relevance quality claim"}})
    validate_dataset(bundle)
    return bundle
