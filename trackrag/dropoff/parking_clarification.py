"""Ask the passenger to clarify an unsupported vehicle reference, without running a model."""
import re
import unicodedata

from . import passenger_parking as parking
from . import vehicle_wording
from .between_road import _between_placement
from .contracts import validate_request
from .inputs import _inspect_image
from .pair_resolution import explicit_pair_quotes
from .passenger import digest, unresolved
from .passenger_pair_choice import OTHER_LOCATION, STRUCTURE, current_tracks, vehicle_head


METHOD = "passenger_vehicle_reference_clarification_v1"
ACTOR = "reference_clarification"
_DETERMINER = re.compile(r"^(?:(?:the|that|this|another|an?|my|our|your|their)\s+)+", re.I)
_LOCATOR = re.compile(r"\s+(?:on\s+(?:the|my|our|your)|at\s+the|by\s+the|near(?:est)?\s+|beside\s+|"
                      r"in front of\s+|just behind\s+|parked\s+|next to\s+|with\s+|facing\s+|further\s+|farther\s+)", re.I)
_CYCLE = re.compile(r"\b(?:bicycles?|e[- ]?bikes?|pedal[- ]cycles?|pushbikes?|bikes?)\b", re.I)
_MOTOR_BIKE = re.compile(r"\b(?:motor|dirt|sport|pit|trail|quad)[ -]+bikes?\b", re.I)
_PLACE = re.compile(r"\b(?:pharmacy|market|supermarket|mall|store|hotel|restaurant|cafe|"
                    r"garage|showroom|dealership|station|school|hospital|mosque|tower|museum|gallery|"
                    r"crossing|intersection|driveway|pavement|sidewalk|park|lane|path|rack|stand|shelter|stop|"
                    r"hydrant|mailbox|postbox|bollard|bench|cone|bin|umbrella|chair|statue|awning|"
                    r"rider|driver|passenger|child|woman|man|lady|gentleman|officer|worker)s?\b", re.I)
_INDEFINITE = re.compile(r"^(?:(?:other|first|second|nearest|closer|further|farther|little|large|big|small|"
                         r"dark|light|" + parking._COLOUR + r")\s+)*(?:one|thing|object|shape)$", re.I)
_DESCRIPTION = re.compile(r"^(?:(?:" + parking._COLOUR + r"|smaller|larger|new|old)\s+)+", re.I)
_DESIGNATOR = re.compile(r"\b(?=[A-Za-z0-9-]*[A-Za-z])(?=[A-Za-z0-9-]*\d)[A-Za-z][A-Za-z0-9-]{1,13}\b")
_NAME = re.compile(r"(?:[A-Z][A-Za-z0-9-]{1,22})(?:\s+[A-Z][A-Za-z0-9-]{1,22}){0,2}$")
_MISSING = re.compile(r"\bbetween\s+(?P<first>[^.,;!?\n]+?)\s+and(?:\s+the)?[\s.,!?]*$", re.I)
_VETO = re.compile(r"\b(?:not|never|except|excluding|without|rather than|instead of|avoid|or)\b|n['’]t\b", re.I)
_CROSSING_TARGET = re.compile(r"\b(?:on|onto|in)\s+(?:(?:the|that|these|those|marked|painted)\s+)*"
                              r"(?:crosswalk|zebra[- ](?:stripes?|crossing)|pedestrian[- ]crossing)\b", re.I)
_LANE_TARGET = re.compile(r"\b(?:in|into|on)\s+(?:(?:the|that|this|active|moving|live|through|traffic|travel|"
                          r"driving|bus|cycle|bike|fast|main|middle|of)\s+)+lane\b|"
                          r"(?:^|[,;]|\bstop|\bout|\boff)\s*(in (?:its|their|his|her) lane)\b|"
                          r"\b(?:in|into) the (?:middle|centre|center) of (?:the |this )?(?:street|road)\b", re.I)
_WORK_AREA_TARGET = re.compile(
    r"\b(?:in|inside|into|within|onto)\s+(?:(?:the|that|this|a|an|active|road)\s+)*"
    r"(?:(?:coned[- ]off|cordoned[- ]off|barricaded|closed)\s+"
    r"(?:(?:construction|work|roadworks?|maintenance)\s+)?(?:area|zone|site|section)|"
    r"(?:construction|roadworks?|work|maintenance)[- ](?:area|zone|site))\b", re.I)
_QUESTIONS = {
    "cycle": "One of those references may be a bicycle. I can check a gap between two motor vehicles. "
             "Could you name both vehicle types and colours, or mark the intended area on the photo?",
    "unclear_type": "I couldn’t identify both vehicle types from those words. Could you describe both by type "
                    "and colour, or mark the intended area on the photo?",
    "missing_reference": "Which other vehicle do you mean? Please name both vehicle types and colours, "
                         "or mark the intended area on the photo.",
    "crossing_target": "You asked to get out on a pedestrian crossing. Could you choose a pull-in before or after it?",
    "lane_target": "You asked to get out in a travel lane. Is there a pull-in off the lane that you meant?",
    "work_area_target": "You asked for a spot inside the work area. Could you choose a pull-in outside it?",
}


def requested_hazard(text):
    """Read an explicitly requested surface, leaving nearby landmarks and negations alone."""
    speech = parking._PREFACE.sub("", text.strip())
    if (_VETO.search(text) or parking._CANCEL.search(text) or not parking._PLACEMENT_ACT.search(speech)):
        return None
    for reason, pattern in (("crossing_target", _CROSSING_TARGET), ("lane_target", _LANE_TARGET),
                            ("work_area_target", _WORK_AREA_TARGET)):
        match = pattern.search(text)
        if match:
            before = re.split(r"[,;.!?]", text[:match.start()])[-1]
            if re.search(r"\b(?:which|who|that['’]s|(?:that|there) (?:is|was)|moving|waiting|parked)\s*$", before, re.I):
                continue
            if reason == "work_area_target" and re.search(
                    r"\b(?:beside|near|by|next to|outside|opposite|past|behind|in front of)\b.+$", before, re.I):
                continue
            span = match.span(1) if match.lastindex else match.span()
            return {"reason":reason,"literal_references":[{"text":text[slice(*span)],"start":span[0],"end":span[1]}]}
    return None


def _kind(phrase):
    head = _DETERMINER.sub("", _LOCATOR.split(phrase, maxsplit=1)[0].strip())
    if not head:
        return "unknown"
    normalized = parking._vehicle_words(head)
    if STRUCTURE.search(head) or OTHER_LOCATION.search(head) or _PLACE.search(head) or normalized == "building":
        return "other"
    if vehicle_head(normalized):
        return "motor"
    if _MOTOR_BIKE.search(head):
        return "other"
    if _CYCLE.search(head):
        return "cycle"
    if _INDEFINITE.fullmatch(head):
        return "unclear_type"
    # A colour plus an unfamiliar proper name, or a model-style alphanumeric designation, warrants
    # a question. Do not classify unrecognized common-noun destinations as vehicles.
    description = _DESCRIPTION.match(head)
    name = head[description.end():] if description else head
    if (_DESIGNATOR.fullmatch(name) or description and _NAME.fullmatch(name)):
        return "unclear_type"
    return "other"


def references(request):
    """Find exactly one identifiable motor-vehicle phrase and one unsupported reference in this turn."""
    text = request["text"]
    frame = request["scene"]["frame"]
    if (request["scene"]["source"] != "predicted"
            or any((v["recording_id"], v["frame_id"]) != (frame["recording_id"], frame["frame_id"])
                   for v in request["dialogue"])):
        return None
    hazard = requested_hazard(text)
    if hazard:
        return hazard
    suggested = vehicle_wording.suggestion(text)
    if suggested is not None:
        start, end = suggested.word_span
        return {"reason":"vehicle_word_check", "literal_references":[{"text":text[start:end],"start":start,"end":end}],
                "suggested_text":suggested.corrected,"question":suggested.question}
    response = vehicle_wording.reply(request)
    if response and response[0] == "replacement_unclear":
        supplied = response[1]
        quoted = supplied if len(supplied) <= 180 else supplied[:177].rstrip() + "…"
        return {"reason":"vehicle_word_replacement_unclear", "literal_references":[],
                "replacement_words":supplied,
                "question":f"You said “{quoted}”. Where is that in the photo? You can also mark the exact spot."}
    if response and response[0] == "unclear":
        return {"reason":"vehicle_word_undetermined", "literal_references":[],
                "question":"Which vehicle or place do you mean? Please describe its colour and position on the photo."}
    if response and response[0] == "declined":
        literal = response[1]
        question = (f"You meant the {literal}. Where is it in the photo? You can also mark the spot." if literal else
                    "Understood. What landmark or place does “card” refer to? You can also mark the spot on the photo.")
        return {"reason":"vehicle_word_declined", "literal_references":[],
                "question":question}
    if (parking.parking_instruction(request)
            or not _between_placement(text) or _VETO.search(text) or parking._CANCEL.search(text)
            or any(c.isalpha() and "LATIN" not in unicodedata.name(c, "") for c in text)):
        return None
    speech = parking._PREFACE.sub("", text.strip())
    if parking._OTHER_ACTION.match(speech) or not (parking._PLACEMENT_ACT.search(speech) or parking._SHOW_AREA.search(speech)):
        return None
    quotes = explicit_pair_quotes(text)
    if quotes:
        kinds = [_kind(q["text"]) for q in quotes]
        if kinds.count("motor") == 1:
            other = 1 - kinds.index("motor")
            if kinds[other] in {"cycle","unclear_type"}:
                return {"reason":kinds[other], "recognized_index":1-other, "literal_references":quotes}
    missing = _MISSING.search(text)
    if missing and _kind(missing["first"]) == "motor" and not re.search(r"\band\b", missing["first"], re.I):
        return {"reason":"missing_reference","recognized_index":0,
                "literal_references":[{"text":missing["first"],"start":missing.start("first"),"end":missing.end("first")}]}
    return None


def clarify(request):
    if request["scene"]["source"] != "predicted":
        return None
    request = validate_request(request)
    evidence = references(request)
    if evidence is None:
        return None
    if _inspect_image(request["scene"]["frame"])[1] != request["scene"]["provenance"]["image_sha256"]:
        raise ValueError("The original image for this clarification changed")
    evidence.update(current_observations_sha256=digest(current_tracks(request)), wording=request["text"])
    provenance = {**parking.input_binding(request), "method":METHOD, "reference_evidence":evidence,
                  "native_model_calls":0, "independent_model_proposal":False}
    question = evidence["question"] if "question" in evidence else _QUESTIONS[evidence["reason"]]
    return {"ok":True, "request_id":request["request_id"], "provenance":provenance,
            "result":unresolved(question)}


def verify(request, outcome):
    reproduced = clarify(request)
    if reproduced is None or reproduced != outcome:
        raise ValueError("The reference clarification does not match the current original image, words and observations")
    return {"input_verified":True,"original_photo_verified":True,"decision_reproduced":True,
            "native_model_calls":0,"independent_model_proposal":False}
