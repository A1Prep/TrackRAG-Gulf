"""Scope the optional parking model to the rider's active two-vehicle instruction."""
import re
import unicodedata

from . import between_road
from .contracts import error_outcome, validate_outcome
from .data import inference_input_sha256
from .passenger import digest
from .passenger_pair_choice import ELLIPTICAL_LOCATION, NEW_LOCATION, OTHER_LOCATION, STRUCTURE, VEHICLE, VEHICLE_PLURAL


METHOD = "identified_vehicle_parking_v1"
ACTOR = "parking_model"
FAILURE_METHOD = "passenger_parking_dispatch_failure"
FAILURES = {"parking_backend_unavailable", "parking_evidence_invalid"}

_PREFACE = re.compile(r"^\s*(?:(?:no|sorry|actually|rather|instead|please|I (?:mean|meant))\b[\s,:;!-]*)+", re.I)
_CANCEL = re.compile(r"\b(?:cancel|never mind|nevermind|forget (?:it|that)|no longer|anywhere else)\b", re.I)
_OTHER_ACTION = re.compile(r"^\s*(?:why|what|who|when|how (?:many|fast|far|long)|is (?:the|that) (?:road|lane))\b", re.I)
_PLACEMENT_ACT = re.compile(
    r"(?:^|[.,:;!?]\s*|\b(?:and|then|so)\s+)\s*(?:(?:please|kindly)\s+)?"
    r"(?:would you mind (?:dropping (?:me|us)(?: off)?|letting (?:me|us) (?:off|out)|getting (?:out|off)"
    r"|pulling (?:over|in|up)|stopping|parking)|(?:(?:(?:can|could|will|would) (?:you|we)|(?:can|could|may) I)\s+(?:please\s+)?"
    r"|(?:I|we) (?:want|need|would like) to\s+|(?:I|we)['’]d like to\s+"
    r"|(?:I|we)['’]ll\s+|is it possible to\s+)?"
    r"(?:drop (?:me|us)(?: off)?|let (?:me|us) (?:off|out)|leave (?:me|us)"
    r"|get (?:out|off)|pull (?:over|in|up)|stop|park|set (?:me|us) down))\b", re.I)
_BARE_AREA = re.compile(
    r"^(?:(?:just|right|over)\s+)?(?:between\b|(?:there[, ]+)?(?:(?:in|at|into|to)\s+)?"
    r"(?:(?:this|the|that|a)\s+)?(?:(?:empty|clear|open|roadside|parking)\s+)?"
    r"(?:gap|opening|space|spot|area|bay)\s+between\b)", re.I)
_SCENE_SPEECH = re.compile(
    r"^(?:is|are|was|were|did|does|do|has|have|had|there(?:['’]s|\s+(?:is|are))"
    r"|(?:I|we) (?:can )?(?:see|saw|notice)|(?:can|could|would|will) you (?:see|tell|check|identify|"
    r"count|find|spot|describe|confirm|show))\b", re.I)
_BARE_OBSERVATION = re.compile(r"\b(?:there (?:is|are)|(?:I|we|you) (?:can see|see|saw|notice)|(?:stands|sits) (?:a|an|the))\b", re.I)
_SHOW_AREA = re.compile(
    r"(?:^|[.,:;!?]\s*|\b(?:and|then|so)\s+)\s*(?:(?:over\s+)?on (?:the |our |your )?(?:left|right)(?: side)?[, ]+)?"
    r"(?:(?:can|could|would|will) you\s+)?(?:please\s+)?(?:show (?:me|us)|mark|highlight|circle|"
    r"point out|indicate|locate)\s+(?:(?:me|us)\s+)?(?:(?:the|this|that|a|an|which)\s+)?"
    r"(?:(?:clearest|clear|nearest|open|empty|vacant|available|marked|unmarked|roadside|curbside|curb|kerb|"
    r"parking|drop.off|road)\s+)*(?:gap|space|spot|strip|bit|opening|area|bay|place|point)\b", re.I)
_NEW_DESTINATION = re.compile(r"\b(?:take|bring|drive)\s+(?:me|us)\s+(?:to|toward)|\b(?:go|head)\s+to\b", re.I)
_OTHER_PLACEMENT = re.compile(r"^(?:(?:(?:drop|let) (?:me|us)(?: off| out)?|stop|park|pull over)\s+)?"
                              r"(?:(?:right|just)\s+)?(?:behind|beyond|before|past|in front of|ahead of|opposite|alongside)\b", re.I)
_REFERENCE = re.compile(r"\b(?:one|ones|both|pair|first|second|nearer|closer|farther|further|leftmost|rightmost)\b", re.I)
_DESCRIPTION = re.compile(r"\b(?:blue|red|yellow|green|white|black|silver|gr[ae]y|cream|dark(?:er)?|light(?:er)?|"
                          r"left(?:most)?|right(?:most)?|nearer|closer|farther|further|nearest|furthest|striped|parked)\b", re.I)
# Use known vehicle head nouns for routing only. The models receive the unmodified names.
_BODY = r"convertible|saloon|roadster|limousine|limo|station wagon|estate car|minibus|jeep|sports car|four.by.four|4x4"
_MARQUE = (r"Alfa Romeo|Aston Martin|Land Rover|Range Rover|Rolls.Royce|Mercedes(?:.Benz)?|"
           r"Toyota|Nissan|Honda|Mazda|Mitsubishi|Isuzu|Subaru|Suzuki|Hyundai|Kia|Genesis|Lexus|Infiniti|Acura|"
           r"Ford|GMC|Chevrolet|Chevy|Cadillac|Buick|Dodge|Chrysler|Ram (?:1500|2500|3500|TRX)|Tesla|Rivian|Lincoln|"
           r"BMW|Audi|Volkswagen|VW|Porsche|Volvo|Peugeot|Renault|Citro[eë]n|Ferrari|Lamborghini|"
           r"Bentley|Jaguar|Fiat|BYD|Geely|Chery|Jetour|GAC|Haval|MG|Camry|Corolla|Land Cruiser|Hilux|Mustang")
_ALIAS = re.compile(r"\b(?:" + _BODY + "|" + _MARQUE + r")(?P<plural>s)?\b", re.I)
_NAMED_VENUE = re.compile(r"\b(?:" + _MARQUE + "|" + _BODY + r"|car|truck|van|SUV|sedan)(?:['’]s)?\s+"
                          r"(?:auto\s+)?(?:dealership|dealer|showroom|garage|store|shop|factory|office|museum|"
                          r"service cent(?:re|er)|sign|logo|billboard|building|entrance|station|bus stop|wash|depot|park|"
                          r"owner|driver|passenger)s?\b", re.I)
_PERSON = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|President)\.?\s+(?:" + _MARQUE + r")\b", re.I)
_COLOUR = r"(?:(?:dark|light)[ -])?(?:black|white|grey|gray|silver|red|blue|green|yellow|orange|brown|beige|cream|gold|purple|pink|maroon)"
_SHARED_HEAD = re.compile(r"\bbetween\s+(?:the\s+)?(?:" + _COLOUR + r")\s+and\s+(?:the\s+)?(?:" + _COLOUR + r")\s+(\w+)\b", re.I)


def _vehicle_words(text):
    protected = _NAMED_VENUE.sub("building", _PERSON.sub("person", text))
    vehicles = _ALIAS.sub(lambda m: "vehicles" if m["plural"] else "vehicle", protected)
    # "The orange and gray SUVs" can refer to two vehicles. The model still sees the full literal words.
    return _SHARED_HEAD.sub(lambda m: "between the two "+m[1] if VEHICLE_PLURAL.fullmatch(m[1]) else m[0], vehicles)


def _qualifier(text):
    """A continuation can identify an earlier vehicle without asking for a fresh destination."""
    words = _PREFACE.sub("", text)
    if (not words or _CANCEL.search(words) or _OTHER_ACTION.match(words) or _SCENE_SPEECH.match(words)
            or _NEW_DESTINATION.search(words)):
        return False
    if (NEW_LOCATION.search(words) or ELLIPTICAL_LOCATION.match(words) or _OTHER_PLACEMENT.match(words)
            or re.search(r"\b(?:behind|in front of)\s+(?:both|them|the pair)\b", words, re.I)):
        return False
    vehicle = VEHICLE.search(words)
    location = OTHER_LOCATION.search(words)
    structure = STRUCTURE.search(words)
    if (location and (not vehicle or location.start() < vehicle.start())
            or structure and (not vehicle or structure.start() <= vehicle.start())):
        return False
    if re.search(r"\bbetween\b", words, re.I):
        return bool(re.fullmatch(r"(?:right |just )?between (?:those two|these two|the two|them(?: both)?)(?:,? please)?[.!? ]*", words, re.I))
    return bool(vehicle or (_REFERENCE.search(words) and _DESCRIPTION.search(words))
                or re.fullmatch(r"(?:both |the ones? )?(?:on (?:the |your |our )?)?(?:left|right)(?: side)?[.!? ]*", words, re.I))


def parking_instruction(request):
    """Return the active literal passenger words, or None for another task or placement.

    Caller supplies a validated predicted-scene request. Only passenger turns on its
    current frame can identify vehicles. An explicit replacement retires the old pair.
    """
    scene = request["scene"]
    if scene["source"] != "predicted":
        return None
    frame = scene["frame"]
    turns = []
    for turn in request["dialogue"]:
        if (turn["recording_id"], turn["frame_id"]) != (frame["recording_id"], frame["frame_id"]):
            return None
        if turn["role"] == "user":
            turns.append(turn["text"])
    turns.append(request["text"])
    active = []
    for text in turns:
        words = text.strip()
        speech = _PREFACE.sub("", words)
        non_latin = any(c.isalpha() and "LATIN" not in unicodedata.name(c, "") for c in words)
        if non_latin or _CANCEL.search(words) or _OTHER_ACTION.match(words):
            active = []
        elif ((_PLACEMENT_ACT.search(speech) or _SHOW_AREA.search(speech)
               or _BARE_AREA.match(speech) and not _BARE_OBSERVATION.search(speech))
              and between_road.requested({"text": _vehicle_words(words), "dialogue": []})):
            active = [text]
        elif active and _qualifier(_vehicle_words(words)):
            active.append(text)
        else:
            active = []
    return "\n".join(active) if active else None


def input_binding(request):
    photo = request["scene"]["provenance"]["image_sha256"]
    return {"request_sha256": digest(request), "image": {"sha256": photo},
            "inference_input_sha256": inference_input_sha256(request, photo)}


def dispatch_failure(request, code, *, rejected=None):
    """Record a failed service call or rejected envelope without manufacturing a model response."""
    messages = {"parking_backend_unavailable": "The parking model could not finish this request.",
                "parking_evidence_invalid": "The parking model’s response could not be verified for this request."}
    source = {**input_binding(request), "method": FAILURE_METHOD, "intended_method": METHOD,
              "independent_model_proposal": False}
    if rejected is not None:
        try:
            source["rejected_response_sha256"] = digest(rejected)
        except (TypeError, ValueError, OverflowError, RecursionError):
            source["rejected_response_serializable"] = False
    return error_outcome(code, messages[code], request["request_id"], source)


def verify_response(request, outcome, *, artifacts=None):
    """Verify the recorded source before showing or confirming it, independent of the transport."""
    checked = validate_outcome(outcome, request)
    source = checked["provenance"]
    binding = input_binding(request)
    if (not parking_instruction(request) or source.get("request_sha256") != binding["request_sha256"]
            or source.get("image", {}).get("sha256") != binding["image"]["sha256"]
            or source.get("inference_input_sha256") != binding["inference_input_sha256"]):
        raise ValueError("Parking response is bound to a different source or passenger instruction")
    if source.get("method") == FAILURE_METHOD:
        code = (checked.get("error") or {}).get("code")
        expected = dispatch_failure(request, code) if code in FAILURES else None
        diagnostic = {k:v for k,v in source.items() if k in {"rejected_response_sha256", "rejected_response_serializable"}}
        if (expected is None or checked["ok"] or checked.get("error") != expected["error"]
                or {k:v for k,v in source.items() if k not in diagnostic} != expected["provenance"]
                or ("rejected_response_sha256" in diagnostic and not re.fullmatch("[0-9a-f]{64}", str(diagnostic["rejected_response_sha256"])))
                or ("rejected_response_serializable" in diagnostic and diagnostic["rejected_response_serializable"] is not False)):
            raise ValueError("Invalid parking dispatch failure")
        from .inputs import _inspect_image
        try:
            original = _inspect_image(request["scene"]["frame"])[1] == binding["image"]["sha256"]
        except (OSError, ValueError):
            original = False
        return {"input_verified": original, "decision_reproduced": False, "independent_model_proposal": False}
    if source.get("method") != METHOD:
        raise ValueError("Expected the registered two-vehicle parking model")
    from .parking_evidence import verify
    proof = verify(request, outcome, artifacts=artifacts)
    proposed = outcome["ok"] and outcome["result"]["status"] == "proposed"
    if (not isinstance(proof, dict) or proof.get("input_verified") is not True or
            proposed and (proof.get("decision_reproduced") is not True or proof.get("independent_model_proposal") is not True)
            or not proposed and proof.get("independent_model_proposal") is True):
        raise ValueError("Parking verification did not establish this decision")
    return proof
