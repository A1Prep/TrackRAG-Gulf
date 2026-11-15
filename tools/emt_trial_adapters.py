"""A selectable baseline for real EMT trials while the task modules are developed.

Reads original human labels, searches descriptions derived from those labels, and
optionally calls Gemini on selected, privacy-processed frames. It supplies no
predicted detector tracks and uses no gold answers in retrieval or generation.
"""

import base64
import io
import json
import math
import os
import re
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

from PIL import Image, ImageDraw

WINDOW_FRAMES = 30
STRIDE_FRAMES = 15


def _read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


class TrialRecording:
    def __init__(self, data_root, recording_id):
        self.directory = Path(data_root) / recording_id
        manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["recording_id"] != recording_id:
            raise ValueError("Prepared recording ID does not match the requested recording")
        self.recording_id = recording_id
        self.frames = {
            row["frame_id"]: dict(row, image_path=str((self.directory / row["image_path"]).absolute()))
            for row in manifest["frames"]
        }

    def iter_frames(self):
        return iter(self.frames[frame] for frame in sorted(self.frames))

    def get_frame(self, frame_id):
        return self.frames[frame_id]


def open_recording(data_root, recording_id):
    if (Path(data_root) / recording_id / "manifest.json").is_file():
        return TrialRecording(data_root, recording_id)
    from trackrag.emt import open_recording as open_emt_recording

    return open_emt_recording(data_root, recording_id)


def load_annotated_tracks(recording):
    if isinstance(recording, TrialRecording):
        return _read_jsonl(recording.directory / "tracks.jsonl")
    from trackrag.emt import load_annotated_tracks as load_emt_tracks

    return load_emt_tracks(recording)


def load_questions(path):
    return _read_jsonl(path)


def build_index(recording, tracks):
    grouped = defaultdict(list)
    for observation in tracks:
        if observation["recording_id"] != recording.recording_id or observation["source"] != "annotated":
            raise ValueError("The trial index accepts one recording's annotated tracks")
        grouped[observation["track_id"]].append(observation)
    return {
        "recording": recording,
        "tracks": {key: sorted(rows, key=lambda row: row["frame_id"]) for key, rows in grouped.items()},
    }


def _description(rows, first_frame, width):
    start, end = rows[0], rows[-1]
    x1, y1, x2, y2 = start["bbox_xyxy"]
    a1, b1, a2, b2 = end["bbox_xyxy"]
    centre = (x1 + x2) / 2 / width
    location = "left" if centre < 1/3 else "right" if centre > 2/3 else "centre"
    delta_x = ((a1 + a2) - (x1 + x2)) / 2 / width
    direction = "right" if delta_x > 0.01 else "left" if delta_x < -0.01 else "stable"
    ratio = ((a2 - a1) * (b2 - b1)) / ((x2 - x1) * (y2 - y1))
    size = "larger growing" if ratio > 1.2 else "smaller shrinking" if ratio < 0.8 else "stable size"
    appearance = "first observed appeared" if start["frame_id"] == first_frame else "continuing"
    vehicle = "vehicle" if start["class_name"].lower() not in {"person", "pedestrian", "cyclist"} else ""
    return f"{start['class_name']} road user {vehicle} {location} image moving {direction} {size} {appearance}"


def _tokens(text):
    synonyms = {"entered": "appeared", "enter": "appeared", "enters": "appeared", "bigger": "larger", "center": "centre"}
    stop = {"the", "a", "an", "which", "is", "was", "it", "on", "of", "in", "as", "does", "during", "this", "and", "to"}
    return Counter(synonyms.get(token, token) for token in re.findall(r"[a-z]+", text.lower()) if token not in stop)


def rank_evidence(question, index, k=3):
    recording = index["recording"]
    if question["recording_id"] != recording.recording_id:
        raise ValueError("Question and search index belong to different recordings")
    first_recording_frame = next(recording.iter_frames())["frame_id"]
    candidates = []
    widths = {}
    for track_id, observations in index["tracks"].items():
        # Generate windows and vocabulary after applying the query cutoff.
        visible = [row for row in observations if row["frame_id"] <= question["query_frame"]]
        if not visible:
            continue
        for start in range(visible[0]["frame_id"], visible[-1]["frame_id"] + 1, STRIDE_FRAMES):
            window = [row for row in visible if start <= row["frame_id"] < start + WINDOW_FRAMES]
            if not window:
                continue
            observed_start = visible[0]["frame_id"]
            appearance_frame = observed_start if observed_start > first_recording_frame else -1
            first_frame = window[0]["frame_id"]
            if first_frame not in widths:
                widths[first_frame] = recording.get_frame(first_frame)["width"]
            text = _description(window, appearance_frame, widths[first_frame])
            candidates.append((track_id, window[0]["frame_id"], window[-1]["frame_id"], _tokens(text)))
    if not candidates:
        return []
    frequency = Counter()
    for _, _, _, tokens in candidates:
        frequency.update(tokens.keys())
    weights = {token: math.log((1 + len(candidates)) / (1 + count)) + 1 for token, count in frequency.items()}
    query = _tokens(question["question"])
    query_norm = math.sqrt(sum((count * weights.get(token, 0)) ** 2 for token, count in query.items()))
    if not query_norm:
        return []
    ranked = []
    for track_id, start, end, tokens in candidates:
        numerator = sum(query[token] * count * weights[token] ** 2 for token, count in tokens.items())
        norm = math.sqrt(sum((count * weights[token]) ** 2 for token, count in tokens.items()))
        score = numerator / (norm * query_norm)
        if score > 0:
            ranked.append((score, track_id, start, end))
    ranked.sort(key=lambda row: (-row[0], row[1], row[2], row[3]))
    return [{
        "recording_id": recording.recording_id, "source": "annotated", "track_id": track_id,
        "start_frame": start, "end_frame": end, "rank": rank, "score": score,
    } for rank, (score, track_id, start, end) in enumerate(ranked[:k], start=1)]


def _hosted_image(path, observations):
    with Image.open(path) as original:
        image = original.convert("RGB")
    draw = ImageDraw.Draw(image)
    people = {"person", "pedestrian", "cyclist", "motorbike"}
    for observation in observations:
        left, top, right, bottom = observation["bbox_xyxy"]
        if observation["class_name"].lower() not in people:
            top += 0.6 * (bottom - top)
        draw.rectangle((left, top, right, bottom), fill=(35, 35, 35))
    image.thumbnail((1024, 576))
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=85)
    return output.getvalue()


def answer_question(question, evidence, run_dir):
    if not evidence:
        return {"answer_status": "insufficient_evidence", "answer": None, "cited_evidence_ids": []}
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY is required for live trial answers")
    model = os.environ.get("TRACKRAG_TRIAL_MODEL", "gemini-3.8-flash")
    observations = _read_jsonl(run_dir / "tracks.jsonl")
    prompt = (
        "Answer the reviewer's question using only the supplied observed frames. "
        "Each image highlights the road user belonging to its evidence ID. "
        "The images show positions in a moving camera's view; do not turn pixel motion into physical speed. "
        "Only state what the images establish. Hidden details or missing measurements require insufficient_evidence. "
        "Identify the evidence IDs supporting every answered result. "
        "Some person regions and lower vehicle regions are masked for privacy. "
        "Return JSON with answer_status (answered or insufficient_evidence), answer (text), cited_evidence_ids (list). "
        "For insufficient_evidence return an empty citation list. "
        f"Question at frame {question['query_frame']}: {question['question']}"
    )
    parts = [{"text": prompt}]
    sent = []
    hosted = run_dir / "hosted" / sha256(question["question_id"].encode()).hexdigest()
    hosted.mkdir(parents=True, exist_ok=True)
    for item in evidence:
        for frame in item["images"]:
            matching = [row for row in observations if row["frame_id"] == frame["frame_id"]]
            payload = _hosted_image(run_dir / frame["image_path"], matching)
            name = f"{item['evidence_id']}-{frame['frame_id']}.jpg"
            (hosted / name).write_bytes(payload)
            parts.extend([
                {"text": f"Evidence {item['evidence_id']}, track {item['interval']['track_id']}, frame {frame['frame_id']}."},
                {"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(payload).decode("ascii")}},
            ])
            sent.append({"evidence_id": item["evidence_id"], "frame_id": frame["frame_id"], "image": name})
    schema = {
        "type": "OBJECT", "required": ["answer_status", "answer", "cited_evidence_ids"],
        "properties": {
            "answer_status": {"type": "STRING", "enum": ["answered", "insufficient_evidence"]},
            "answer": {"type": "STRING"},
            "cited_evidence_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
        },
    }
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=json.dumps({"contents": [{"role": "user", "parts": parts}], "generationConfig": {
            "temperature": 0, "maxOutputTokens": 2048, "responseMimeType": "application/json", "responseSchema": schema,
        }}).encode(), headers={"Content-Type": "application/json", "x-goog-api-key": key},
    )
    receipt = {"question_id": question["question_id"], "model": model, "sent": sent, "request_status": "pending"}
    receipt_path = hosted / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            result = json.load(response)
        contents = result["candidates"][0]["content"]["parts"]
        answer = json.loads("".join(part["text"] for part in contents if "text" in part and not part.get("thought")))
    except Exception as error:
        receipt.update(request_status="error", error_type=type(error).__name__)
        if isinstance(error, urllib.error.HTTPError):
            receipt["http_status"] = error.code
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        if isinstance(error, urllib.error.HTTPError):
            raise RuntimeError(f"Gemini {model} returned HTTP {error.code}") from error
        raise
    receipt.update(request_status="completed", usage=result.get("usageMetadata"), answer=answer)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return answer
