"""Resolve the passenger's visual reference before estimating a destination region."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re

from .contracts import validate_box, validate_request
from .model import GroundingConfig, _image_snapshot, _materialize_image, verify_adapter
from .structured import EMPTY_LANDMARK_CONTRACT, EMPTY_LANDMARK_PATTERN, LANDMARK_CONTRACT


GUIDE = """Look at the current front-facing road photograph. Interpret the passenger's final message, using the earlier dialogue for a correction or clarification. First identify the VISUAL LANDMARK(S) used to describe where they want to get out. Road-user detector boxes are possible landmarks; their coordinates are x1 y1 x2 y2 on a 0..1000 scale. Their names describe broad object types and can be imperfect. Read colour and appearance from the image.
Reply with exactly one line in one of these forms:
LANDMARK L3 (or LANDMARK L2 L6 if both particular landmarks are needed). Use actual listed identifiers.
OBJECT x1 y1 x2 y2 for a clearly visible landmark without a matching listed box, including a fence, entrance or marked surface. Report the visible reference itself here. A location later estimated relative to it is a separate step.
ASK followed by one short distinguishing question when several plausible places/objects remain or a gesture such as 'here' is missing.
MISSING when the described reference cannot be seen.
Use the requested side, count, colour, order and negation. A request between two objects requires BOTH identities. Ask about the missing second object if just one matches. Broadly applied phrases like 'here' need a visible distinguishing reference from the passenger. If the relation to a specific visible object is approximate (e.g. 'beside the bus'), identifying that object suffices for a proposal that the passenger will check. Give only the landmark decision. Do not predict a stopping point or an AREA in this step."""

EMPTY_GUIDE = """Look at the current road photograph and any earlier dialogue. Identify the visible reference the passenger uses to describe the desired destination. There are no current detector boxes. Inspect the original pixels, including structures, entrances and road users.
Reply with exactly one line from these three forms:
OBJECT x1 y1 x2 y2 for one clearly identified visible reference. Use four whole numbers from 0 to 1000, scaled independently against the image width and height, from the top-left corner to the bottom-right corner of the reference itself.
ASK followed by one short English question ending in a question mark if the passenger could mean several places, refers to a gesture you cannot see, or needs to distinguish two references.
MISSING if the requested visual reference cannot be found.
Use the passenger's side, colours, count, order, negation and corrections. Only describe the reference. The destination area will be estimated separately."""


def resolve_image(request, config):
    frame = request["scene"]["frame"]
    image, record = _image_snapshot(frame, config)
    expected = request["scene"].get("provenance", {}).get("image_sha256")
    if not isinstance(expected, str) or expected != record["sha256"]:
        image.close()
        raise ValueError("Passenger scene requires a matching original-image SHA256")
    return image, record


def landmark_messages(request):
    request = validate_request(request)
    frame = request["scene"]["frame"]
    w, h = frame["width"], frame["height"]
    current = sorted((t for t in request["scene"]["tracks"] if t["frame_id"] == frame["frame_id"]),
                     key=lambda t: (t["bbox_xyxy"][0], t["bbox_xyxy"][1], t["track_id"]))
    index = {f"L{i}": t for i, t in enumerate(current, 1)}
    listed = []
    for identity, track in index.items():
        box = [round(1000*v/d) for v, d in zip(track["bbox_xyxy"], (w, h, w, h))]
        listed.append(f"{identity} {track['class_name']} " + " ".join(map(str, box)))
    image_context = ("Current detector candidates:\n" + "\n".join(listed) if index else
                     "Original road photograph. No current detector boxes are available.")
    messages = [{"role": "system", "content": GUIDE if index else EMPTY_GUIDE},
                {"role": "user", "content": [{"type": "image", "image": frame["image_path"]},
                 {"type": "text", "text": image_context}]}]
    messages.extend({"role": t["role"], "content": t["text"]} for t in request["dialogue"])
    messages.append({"role": "user", "content": request["text"]})
    return messages, index


class InvalidLandmarkIDs(ValueError):
    message = "Landmarks must refer to one or two distinct current detector IDs"

    def __init__(self):
        super().__init__(self.message)


def parse_landmark(text, index, frame):
    if not isinstance(text, str) or not 1 <= len(text.strip()) <= 1000:
        raise ValueError("Landmark response must be a bounded, nonempty string")
    line = text.strip()
    if not index and re.fullmatch(EMPTY_LANDMARK_PATTERN, line) is None:
        raise ValueError("With no current detector boxes the model must return a complete OBJECT, ASK or MISSING response")
    if line == "MISSING":
        return {"kind": "missing"}
    if line.startswith("ASK ") and len(line[4:].strip()) > 2 and "\n" not in line:
        return {"kind": "ask", "question": line[4:].strip()}
    parts = line.split()
    if not parts:
        raise ValueError("Empty landmark response")
    if all(re.fullmatch(r"L[1-9][0-9]*", p) for p in parts):
        parts.insert(0, "LANDMARK")
    if parts[0] == "LANDMARK":
        ids = parts[1:]
        if len(ids) not in (1, 2) or len(set(ids)) != len(ids) or any(i not in index for i in ids):
            raise InvalidLandmarkIDs()
        return {"kind": "landmark", "track_ids": [index[i]["track_id"] for i in ids],
                "boxes": [deepcopy(index[i]["bbox_xyxy"]) for i in ids]}
    if parts[0] == "OBJECT" and len(parts) == 5:
        if any(re.fullmatch(r"[0-9]{1,4}(?:\.[0-9]+)?", s) is None for s in parts[1:]):
            raise ValueError("Expected four normalized box coordinates")
        coords = validate_box([float(v) for v in parts[1:]], 1000, 1000)
        box = [v*d/1000 for v, d in zip(coords, (frame["width"], frame["height"])*2)]
        return {"kind": parts[0].lower(), "boxes": [box], "track_ids": []}
    raise ValueError("Unrecognized landmark response")


class LandmarkResolver:
    def __init__(self, config):
        self.config = config if isinstance(config, GroundingConfig) else GroundingConfig.from_mapping(config)
        if self.config.runtime != "streamed" or self.config.allow_annotated:
            raise ValueError("Passenger landmark inference uses the local streamed predicted-input condition")
        self.backend = None
        self.adapter = None

    def resolve(self, request, image):
        from .streamed import StreamedProcessBackend
        messages, index = landmark_messages(request)
        contract = LANDMARK_CONTRACT if index else EMPTY_LANDMARK_CONTRACT
        record = {"messages": messages, "model_id": self.config.model_id,
                  "revision": self.config.revision, "contract": contract,
                  "messages_sha256": hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()}
        record["request_sha256"] = hashlib.sha256(json.dumps(request, sort_keys=True,
            separators=(",",":"), allow_nan=False).encode()).hexdigest()
        record["image_sha256"] = request["scene"]["provenance"]["image_sha256"]
        current = verify_adapter(self.config)
        if self.backend is None:
            self.backend = StreamedProcessBackend(self.config)
            self.adapter = current
        if current != self.adapter or verify_adapter(self.config) != self.adapter:
            self.close()
            raise ValueError("Landmark adapter changed during loading or between requests")
        record["adapter"] = self.adapter
        supplied = _materialize_image(messages, image)
        try:
            response = self.backend.generate(supplied, config=self.config, contract=contract)
        except Exception:
            self.close()
            raise
        record.update(raw_text=response.text, backend_details=response.details)
        try:
            record["decision"] = parse_landmark(response.text, index, request["scene"]["frame"])
        except ValueError as error:
            record["parse_error"] = str(error)
        return record

    def close(self):
        if self.backend is not None:
            self.backend.close()
            self.backend = None
