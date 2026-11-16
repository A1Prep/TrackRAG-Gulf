"""Validate saved reviewer artifacts without importing or calling model adapters."""

import json
import math
from hashlib import sha256
from pathlib import Path, PureWindowsPath

from PIL import Image

from .score import read_jsonl


def fingerprint(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def validate_question(question):
    if set(question) != {"question_id", "recording_id", "query_frame", "question"}:
        raise ValueError("Questions require the four public fields, without answer-key fields")
    for key in ("question_id", "recording_id", "question"):
        if not isinstance(question[key], str) or not question[key].strip():
            raise ValueError(f"Question {key} must be a nonempty string")
    integer(question["query_frame"], "query_frame")


def local_asset(directory, relative):
    if not isinstance(relative, str) or not relative or PureWindowsPath(relative).drive:
        raise ValueError("Evidence image paths must be relative to their run directory")
    path = (directory / relative.replace("\\", "/")).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError(f"Evidence asset is missing or outside its run: {relative}")
    return path


def _box(box):
    if (not isinstance(box, (list, tuple)) or len(box) != 4
            or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in box)
            or not 0 <= box[0] < box[2] or not 0 <= box[1] < box[3]):
        raise ValueError("Observation boxes require four finite original-pixel coordinates and positive area")


def audit_run(directory, expected):
    """Audit stored consistency, not image authenticity or model factual accuracy."""
    directory = Path(directory).resolve()
    metadata = read_object(directory / "run.json")
    recording_id, source = metadata["recording_id"], metadata["source"]
    if (not isinstance(recording_id, str) or not recording_id.strip()
            or recording_id not in {q["recording_id"] for q in expected.values()}):
        raise ValueError("Run recording is outside the shared question set")
    if source not in {"annotated", "predicted"}:
        raise ValueError("Run source must be annotated or predicted")
    rows = read_jsonl(directory / "results.jsonl")
    tracks = read_jsonl(directory / "tracks.jsonl")
    if integer(metadata["question_count"], "question_count") != len(rows):
        raise ValueError("Run metadata question_count disagrees with its result file")
    if integer(metadata["observation_count"], "observation_count") != len(tracks):
        raise ValueError("Run metadata observation_count disagrees with its track file")
    observations = {}
    for row in tracks:
        if row["recording_id"] != recording_id or row["source"] != source:
            raise ValueError("Track file mixes recordings or sources")
        key = (integer(row["track_id"], "track_id"), integer(row["frame_id"], "frame_id"))
        if key in observations:
            raise ValueError("Duplicate track/frame observation")
        _box(row["bbox_xyxy"])
        observations[key] = row
    results, files, image_sizes = {}, {}, {}

    def record_file(path, role):
        path = Path(path).resolve()
        if str(path) not in files:
            files[str(path)] = {"path": str(path), "role": role, "sha256": fingerprint(path)}

    def verify_image(path):
        if path not in image_sizes:
            with Image.open(path) as image:
                image_sizes[path] = image.size
                image.verify()
        return image_sizes[path]

    for name in ("run.json", "results.jsonl", "tracks.jsonl"):
        record_file(directory / name, name)
    receipt_rows = []
    for row in rows:
        question = row["question"]
        validate_question(question)
        qid = question["question_id"]
        if qid not in expected or question != expected[qid]:
            raise ValueError(f"Question text, recording or cutoff differs from the shared question set: {qid}")
        if qid in results:
            raise ValueError(f"Duplicate result question: {qid}")
        if question["recording_id"] != recording_id or row["source"] != source:
            raise ValueError("Result mixes recordings or sources")
        evidence = row["evidence"]
        if not isinstance(evidence, list) or len(evidence) > 3:
            raise ValueError("Expected at most three evidence intervals")
        ids, image_pairs = set(), set()
        for rank, item in enumerate(evidence, 1):
            eid = item["evidence_id"]
            if not isinstance(eid, str) or not eid.strip() or eid in ids:
                raise ValueError("Evidence IDs must be nonempty and unique within a question")
            ids.add(eid)
            interval = item["interval"]
            start = integer(interval["start_frame"], "start_frame")
            end = integer(interval["end_frame"], "end_frame")
            if start > end or end > question["query_frame"]:
                raise ValueError(f"Reversed interval or future evidence: {qid}")
            if interval["recording_id"] != recording_id or interval["source"] != source:
                raise ValueError("Evidence mixes recordings or sources")
            if type(interval["rank"]) is not int or interval["rank"] != rank:
                raise ValueError("Evidence ranks must be consecutive from one")
            score = interval["score"]
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError("Evidence scores must be finite")
            track_id = interval["track_id"]
            if track_id is None:
                if start != end:
                    raise ValueError("Scene evidence requires one exact frame")
            else:
                integer(track_id, "track_id")
            images = item["images"]
            if not isinstance(images, list) or not 1 <= len(images) <= 3:
                raise ValueError("Each evidence interval requires one to three saved images")
            seen_frames = set()
            for image in images:
                frame = integer(image["frame_id"], "image frame_id")
                if not start <= frame <= end or frame in seen_frames:
                    raise ValueError("Evidence image is outside its interval or duplicated")
                seen_frames.add(frame)
                box = image["bbox_xyxy"]
                if track_id is None:
                    if box is not None:
                        raise ValueError("Scene images cannot invent a vehicle box")
                elif (track_id, frame) not in observations or box != observations[(track_id, frame)]["bbox_xyxy"]:
                    raise ValueError("Evidence box does not match the saved track observation")
                path = local_asset(directory, image["image_path"])
                width, height = verify_image(path)
                if box is not None and (box[2] > width or box[3] > height):
                    raise ValueError("Evidence box exceeds its image dimensions")
                record_file(path, "evidence_image")
                image_pairs.add((eid, frame))
        answer = row["answer"]
        status, text, citations = answer["answer_status"], answer["answer"], answer["cited_evidence_ids"]
        if status not in {"answered", "insufficient_evidence", "error", "not_requested"}:
            raise ValueError("Unknown answer status")
        if (not isinstance(citations, list) or any(not isinstance(c, str) or c not in ids for c in citations)
                or len(set(citations)) != len(citations)):
            raise ValueError("Answer contains invalid or duplicate citations")
        if text is not None and not isinstance(text, str):
            raise ValueError("Answer text must be a string or null")
        if status == "answered" and (not isinstance(text, str) or not text.strip() or not citations):
            raise ValueError("An answered result requires text and supplied citations")
        if status != "answered" and citations:
            raise ValueError("An unanswered result cannot claim supporting citations")
        hosted = directory / "hosted" / sha256(qid.encode()).hexdigest()
        for name in ("draft-receipt.json", "receipt.json"):
            path = hosted / name
            if not path.exists():
                continue
            path = local_asset(directory, path.relative_to(directory).as_posix())
            receipt = read_object(path)
            if receipt.get("question_id", qid) != qid:
                raise ValueError("Provider receipt belongs to a different question")
            if name == "receipt.json":
                if receipt.get("request_status") == "error" and status != "error":
                    raise ValueError("Delivered answer contradicts its failed final provider receipt")
                response = receipt.get("response")
                if not isinstance(response, dict):
                    response = receipt.get("answer")  # Earlier TF-IDF trial format.
                if (receipt.get("request_status") == "completed" and status != "error"
                        and isinstance(response, dict) and "answer_status" in response and "cited_evidence_ids" in response):
                    supplied = response["cited_evidence_ids"]
                    if (response["answer_status"] != status or not isinstance(supplied, list)
                            or any(not isinstance(c, str) for c in supplied) or set(supplied) != set(citations)):
                        raise ValueError("Delivered status or citations contradict the final provider response")
            for sent in receipt.get("sent", []):
                frame = integer(sent["frame_id"], "provider frame_id")
                if frame > question["query_frame"] or (sent["evidence_id"], frame) not in image_pairs:
                    raise ValueError("Provider receipt references future or unsupplied evidence")
                processed = local_asset(hosted, sent["image"])
                verify_image(processed)
                record_file(processed, "provider_input")
            record_file(path, "provider_receipt")
            receipt_rows.append({key: receipt.get(key) for key in ("model", "reported_model", "role", "request_status", "cached")})
        results[qid] = row
    config = metadata.get("configuration", {})
    if not isinstance(config, dict):
        raise ValueError("Run configuration must be an object")
    return {
        "directory": directory, "recording_id": recording_id, "source": source,
        "results": results, "tracks": tracks, "files": list(files.values()),
        "provenance": {
            "directory": str(directory),
            "recording_id": recording_id, "source": source,
            "recording_adapter": metadata.get("recording_adapter"),
            "adapter_module": config.get("adapter_module"), "answerer": config.get("answerer"),
            "provider_receipts": receipt_rows,
        },
    }
