"""Shared, annotation-free inputs and separate semantic/operational outcomes.

Coordinates at these boundaries are original-image pixels. Model-internal
normalization is handled by formatting.py, not by the scene provider or UI.
"""

from copy import deepcopy
from hashlib import sha256
import json
import math
from numbers import Real
from typing import Literal, NotRequired, Protocol, TypedDict

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
    session_id: NotRequired[str]  # App sessions supply this; existing batch inputs remain valid.


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


class GroundingError(TypedDict):
    code: str
    message: str


class GroundingFailure(TypedDict):
    ok: Literal[False]
    request_id: str
    error: GroundingError
    provenance: dict


GroundingOutcome = GroundingSuccess | GroundingFailure


class GroundingAdapter(Protocol):
    """Task 5 supplies the implementation; Task 4 consumes this boundary."""

    def interpret(self, request: InstructionRequest) -> GroundingOutcome:
        ...


class SceneReference(TypedDict):
    recording_id: str
    frame_id: int
    source: TrackSource


class DestinationProposal(TypedDict):
    """An app-assigned identity for one immutable, validated model proposal."""

    proposal_id: str
    request_id: str
    scene: SceneReference
    result: GroundingResult
    provenance: dict


class ProposalConfirmation(TypedDict):
    """Created only after explicit passenger confirmation in the application."""

    proposal_id: str
    request_id: str
    scene: SceneReference
    proposal_sha256: str


class TranscriptCorrection(TypedDict):
    request_id: str
    original_transcript: str
    corrected_text: str


class InteractionState(TypedDict):
    """Snapshot for a submitted instruction; transitions belong to Task 4's controller."""

    session_id: str
    request: InstructionRequest
    status: Literal[
        "pending", "needs_clarification", "insufficient_visual_evidence",
        "proposed", "confirmed", "cancelled", "error",
    ]
    pending_clarification: str | None
    proposal: DestinationProposal | None
    confirmation: ProposalConfirmation | None
    transcript_corrections: list[TranscriptCorrection]


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
          ("original_transcript", "session_id"), "request")
    _text(request["request_id"], "request_id", 200)
    if "session_id" in request:
        _text(request["session_id"], "session_id", 200)
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


def _json_copy(value, name):
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain only finite JSON values") from error
    if len(encoded.encode("utf-8")) > 2_000_000:
        raise ValueError(f"{name} exceeds the serialized record budget")
    return deepcopy(value)


def validate_outcome(outcome, request, *, allow_annotated=False):
    """Check an adapter envelope against the exact request before the app uses it.

    Rejection raises ValueError: the caller retains an invalid_model_output error
    receipt, never a semantic abstention. Existing model wire fields are unchanged.
    """
    request = validate_request(request, allow_annotated=allow_annotated)
    if not isinstance(outcome, dict) or type(outcome.get("ok")) is not bool:
        raise ValueError("outcome ok must be a boolean")
    payload = "result" if outcome["ok"] else "error"
    _keys(outcome, ("ok", "request_id", payload, "provenance"), name="outcome")
    if outcome["request_id"] != request["request_id"]:
        raise ValueError("outcome belongs to a different/stale request")
    if not isinstance(outcome["provenance"], dict):
        raise ValueError("outcome provenance must be an object")
    checked = _json_copy(outcome, "outcome")
    if outcome["ok"]:
        result = outcome["result"]
        _keys(result, FULL_FIELDS, ("anchor_boxes_xyxy",), "grounding result")
        checked["result"] = validate_target(result, list(result), request["scene"]["frame"])
    else:
        _keys(outcome["error"], ("code", "message"), name="grounding error")
        _text(outcome["error"]["code"], "error code", 100)
        _text(outcome["error"]["message"], "error message", 4096)
    return checked


def _scene_reference(request):
    frame = request["scene"]["frame"]
    return {"recording_id": frame["recording_id"], "frame_id": frame["frame_id"],
            "source": request["scene"]["source"]}


def _validate_scene_reference(reference, request):
    _keys(reference, ("recording_id", "frame_id", "source"), name="scene reference")
    _integer(reference["frame_id"], "scene reference frame_id")
    if reference != _scene_reference(request):
        raise ValueError("scene reference belongs to a different recording/source/frame")


def validate_proposal(proposal, request, *, allow_annotated=False):
    """Validate an app-owned proposal; unresolved/error outcomes cannot be promoted."""
    request = validate_request(request, allow_annotated=allow_annotated)
    _keys(proposal, ("proposal_id", "request_id", "scene", "result", "provenance"),
          name="proposal")
    _text(proposal["proposal_id"], "proposal_id", 200)
    _validate_scene_reference(proposal["scene"], request)
    outcome = validate_outcome(
        {"ok": True, "request_id": proposal["request_id"], "result": proposal["result"],
         "provenance": proposal["provenance"]}, request, allow_annotated=allow_annotated,
    )
    if outcome["result"]["status"] != "proposed":
        raise ValueError("a proposal requires a proposed semantic result")
    checked = deepcopy(proposal)
    checked["result"] = outcome["result"]
    return checked


def proposal_fingerprint(proposal, request, *, allow_annotated=False):
    """Bind confirmation to the proposal content and exact input, including corrections.

    This is a consistency check, not authentication or a confirmation event.
    The scene adapter's image_sha256 additionally binds the original image bytes.
    """
    request = validate_request(request, allow_annotated=allow_annotated)
    proposal = validate_proposal(proposal, request, allow_annotated=allow_annotated)
    content = {"request": request, "proposal": proposal}
    return sha256(json.dumps(content, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")).hexdigest()


def validate_interaction_state(state, *, allow_annotated=False):
    """Validate a snapshot, including stale confirmation and transcript boundaries.

    This does not generate IDs, call a model, perform a transition or confirm on
    the passenger's behalf. The controller must retain every call outcome and
    clear old proposals/confirmations when a new request or scene is selected.
    """
    _keys(state, ("session_id", "request", "status", "pending_clarification", "proposal",
                  "confirmation", "transcript_corrections"), name="interaction state")
    _text(state["session_id"], "session_id", 200)
    request = validate_request(state["request"], allow_annotated=allow_annotated)
    if request.get("session_id", state["session_id"]) != state["session_id"]:
        raise ValueError("request belongs to a different session")
    statuses = ("pending", "needs_clarification", "insufficient_visual_evidence",
                "proposed", "confirmed", "cancelled", "error")
    if state["status"] not in statuses:
        raise ValueError("invalid interaction status")
    if state["status"] == "needs_clarification":
        _text(state["pending_clarification"], "pending_clarification", 2048)
    elif state["pending_clarification"] is not None:
        raise ValueError("pending clarification requires needs_clarification status")
    checked = _json_copy(state, "interaction state")
    if state["status"] in ("proposed", "confirmed"):
        checked["proposal"] = validate_proposal(state["proposal"], request,
                                                allow_annotated=allow_annotated)
    elif state["proposal"] is not None:
        raise ValueError("only proposed/confirmed states can contain a current proposal")
    if state["status"] == "confirmed":
        confirmation = state["confirmation"]
        _keys(confirmation, ("proposal_id", "request_id", "scene", "proposal_sha256"),
              name="confirmation")
        _validate_scene_reference(confirmation["scene"], request)
        proposal = checked["proposal"]
        if (confirmation["proposal_id"] != proposal["proposal_id"] or
                confirmation["request_id"] != request["request_id"] or
                confirmation["proposal_sha256"] != proposal_fingerprint(
                    proposal, request, allow_annotated=allow_annotated)):
            raise ValueError("confirmation is stale or bound to a different proposal/request")
    elif state["confirmation"] is not None:
        raise ValueError("only confirmed status can contain a confirmation")
    corrections = state["transcript_corrections"]
    if not isinstance(corrections, list) or len(corrections) > 32:
        raise ValueError("transcript corrections must be a list of at most 32 records")
    if request["input_mode"] == "voice" and "original_transcript" not in request:
        raise ValueError("voice interaction must preserve the original transcript")
    for correction in corrections:
        _keys(correction, ("request_id", "original_transcript", "corrected_text"),
              name="transcript correction")
        _text(correction["request_id"], "correction request_id", 200)
        original = correction["original_transcript"]
        if not isinstance(original, str) or len(original) > 8192:
            raise ValueError("original transcript must be bounded text")
        _text(correction["corrected_text"], "corrected_text")
        if correction["request_id"] == request["request_id"] and (
            request["input_mode"] != "voice" or original != request.get("original_transcript") or
            correction["corrected_text"] != request["text"]
        ):
            raise ValueError("current correction differs from the voice request")
    return checked
