"""Shared, annotation-free inputs and separate semantic/operational outcomes.

Coordinates at these boundaries are original-image pixels. Model-internal
normalization is handled by formatting.py, not by the scene provider or UI.
"""

from copy import deepcopy
import json
import math
from numbers import Real
from typing import Literal, NotRequired, TypedDict

from trackrag.contracts import Frame, TrackObservation, TrackSource


class SceneContext(TypedDict):
    frame: Frame
    source: TrackSource
    tracks: list[TrackObservation]
    provenance: NotRequired[dict]


class DialogueTurn(TypedDict):
    role: Literal["user", "assistant"]
    text: str
    recording_id: str
    frame_id: int


class InstructionRequest(TypedDict):
    request_id: str
    scene: SceneContext
    text: str
    input_mode: Literal["typed", "voice"]
    dialogue: list[DialogueTurn]
    original_transcript: NotRequired[str]


class GroundingResult(TypedDict):
    status: Literal["proposed", "needs_clarification", "insufficient_visual_evidence"]
    destination_point_xy: list[float] | None
    destination_box_xyxy: list[float] | None
    clarification_question: str | None
    anchor_boxes_xyxy: NotRequired[list[list[float]] | None]


class GroundingSuccess(TypedDict):
    ok: Literal[True]
    request_id: str
    result: GroundingResult
    provenance: dict


class GroundingFailure(TypedDict):
    ok: Literal[False]
    request_id: str
    error: dict[str, str]
    provenance: dict


GroundingOutcome = GroundingSuccess | GroundingFailure


STATUSES = frozenset(("proposed", "needs_clarification", "insufficient_visual_evidence"))
TARGET_FIELDS = frozenset((
    "status", "destination_point_xy", "destination_box_xyxy",
    "clarification_question", "anchor_boxes_xyxy",
))
FULL_FIELDS = frozenset((
    "status", "destination_point_xy", "destination_box_xyxy", "clarification_question",
))
DESTINATION_FIELDS = frozenset(("destination_point_xy", "destination_box_xyxy"))


def _keys(value, required, optional=(), name="record"):
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    missing = set(required) - value.keys()
    extra = value.keys() - set(required) - set(optional)
    if missing or extra:
        raise ValueError(f"{name} has missing {sorted(missing)} or unknown {sorted(extra)} fields")


def _text(value, name, maximum=8192):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{name} must be nonempty text of at most {maximum} characters")
    return value


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def validate_box(value, width, height, name="box"):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{name} must have four coordinates")
    left, top, right, bottom = [_number(v, name) for v in value]
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError(f"{name} must be nonempty and inside the original image")
    return [left, top, right, bottom]


def validate_point(value, width, height, name="point"):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must have two coordinates")
    x, y = [_number(v, name) for v in value]
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"{name} must be inside the original image")
    return [x, y]


def validate_request(request, *, allow_annotated=False):
    """Return a deep copy, rejecting gold fields, future tracks and stale dialogue.

    Structural validation does not open images. The dataset/model adapter must
    verify the referenced bytes and dimensions before using them.
    """
    _keys(request, ("request_id", "scene", "text", "input_mode", "dialogue"),
          ("original_transcript",), "request")
    _text(request["request_id"], "request_id", 200)
    _text(request["text"], "text")
    if request["input_mode"] not in ("typed", "voice"):
        raise ValueError("input_mode must be typed or voice")
    if "original_transcript" in request:
        if not isinstance(request["original_transcript"], str) or len(request["original_transcript"]) > 8192:
            raise ValueError("original_transcript must be bounded text, including empty failed-ASR text")
    scene = request["scene"]
    _keys(scene, ("frame", "source", "tracks"), ("provenance",), "scene")
    frame = scene["frame"]
    _keys(frame, ("recording_id", "frame_id", "timestamp_s", "width", "height", "image_path"),
          name="frame")
    _text(frame["recording_id"], "recording_id", 200)
    _integer(frame["frame_id"], "frame_id")
    if _number(frame["timestamp_s"], "timestamp_s") < 0:
        raise ValueError("timestamp_s cannot be negative")
    width = _integer(frame["width"], "width", 1)
    height = _integer(frame["height"], "height", 1)
    _text(frame["image_path"], "image_path", 4096)
    if scene["source"] not in ("predicted", "annotated"):
        raise ValueError("unknown track source")
    if scene["source"] == "annotated" and not allow_annotated:
        raise ValueError("annotated scene inputs require an explicit oracle/development condition")
    if "provenance" in scene and not isinstance(scene["provenance"], dict):
        raise ValueError("scene provenance must be an object")
    tracks = scene["tracks"]
    if not isinstance(tracks, list) or len(tracks) > 512:
        raise ValueError("scene tracks must be a list of at most 512 observations")
    seen = set()
    for track in tracks:
        _keys(track, ("recording_id", "frame_id", "track_id", "class_name", "bbox_xyxy", "source"),
              name="track")
        if track["recording_id"] != frame["recording_id"] or track["source"] != scene["source"]:
            raise ValueError("track recording/source differs from the scene")
        observed = _integer(track["frame_id"], "track frame_id")
        if observed > frame["frame_id"]:
            raise ValueError("future track observation is not permitted")
        identity = _integer(track["track_id"], "track_id")
        key = (observed, identity)
        if key in seen:
            raise ValueError("duplicate observation for a frame and track")
        seen.add(key)
        _text(track["class_name"], "class_name", 100)
        validate_box(track["bbox_xyxy"], width, height, "track bbox")
    dialogue = request["dialogue"]
    if not isinstance(dialogue, list) or len(dialogue) > 16:
        raise ValueError("dialogue must contain at most 16 turns")
    for turn in dialogue:
        _keys(turn, ("role", "text", "recording_id", "frame_id"), name="dialogue turn")
        if turn["role"] not in ("user", "assistant"):
            raise ValueError("dialogue role must be user or assistant")
        _text(turn["text"], "dialogue text", 4096)
        _integer(turn["frame_id"], "dialogue frame_id")
        if turn["recording_id"] != frame["recording_id"] or turn["frame_id"] != frame["frame_id"]:
            raise ValueError("dialogue is bound to a different/stale scene")
    try:
        encoded = json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("request must contain only finite JSON values") from error
    if len(encoded.encode("utf-8")) > 2_000_000:
        raise ValueError("request exceeds the serialized input budget")
    return deepcopy(request)


def validate_target(target, supervised_fields, frame):
    """Validate explicitly supplied original-pixel labels; never fill missing labels."""
    if not isinstance(supervised_fields, list) or not all(isinstance(x, str) for x in supervised_fields):
        raise ValueError("supervised_fields must be a list of field names")
    if len(set(supervised_fields)) != len(supervised_fields):
        raise ValueError("duplicate supervised field")
    fields = set(supervised_fields)
    if not isinstance(target, dict) or set(target) != fields or not fields <= TARGET_FIELDS:
        raise ValueError("target fields must exactly match the declared available supervision")
    destination_only = fields == DESTINATION_FIELDS
    if not destination_only and not FULL_FIELDS <= fields:
        raise ValueError("supported targets are destination-only or a complete interpretation")
    width, height = frame["width"], frame["height"]
    result = deepcopy(target)
    if not destination_only and (not isinstance(target["status"], str) or target["status"] not in STATUSES):
        raise ValueError("invalid semantic status")
    proposed = destination_only or target["status"] == "proposed"
    if proposed:
        point = validate_point(target["destination_point_xy"], width, height)
        box = validate_box(target["destination_box_xyxy"], width, height)
        if not (box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]):
            raise ValueError("destination point lies outside its proposed coarse region")
        result["destination_point_xy"], result["destination_box_xyxy"] = point, box
    elif target["destination_point_xy"] is not None or target["destination_box_xyxy"] is not None:
        raise ValueError("an unresolved interpretation cannot contain a destination proposal")
    if not destination_only:
        question = target["clarification_question"]
        if target["status"] == "needs_clarification":
            _text(question, "clarification_question", 2048)
        elif question is not None:
            raise ValueError("clarification_question must be null unless clarification is needed")
    if "anchor_boxes_xyxy" in fields:
        anchors = target["anchor_boxes_xyxy"]
        if anchors is not None:
            if not isinstance(anchors, list) or len(anchors) > 16:
                raise ValueError("anchor boxes must be null or a list of at most 16 boxes")
            result["anchor_boxes_xyxy"] = [validate_box(box, width, height, "anchor box") for box in anchors]
    return result


def error_outcome(code, message, request_id, provenance=None):
    """Operational failures are not learned uncertainty or valid abstention."""
    _text(code, "error code", 100)
    _text(message, "error message", 4096)
    return {
        "ok": False, "request_id": request_id,
        "error": {"code": code, "message": message},
        "provenance": deepcopy(provenance) if provenance is not None else {},
    }
