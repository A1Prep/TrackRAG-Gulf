"""Separate a photographed arrangement from unmeasured distances to road landmarks."""
import re


_VEHICLE = r"(?:trucks?|lorry|lorries|pickups?|cars?|sedans?|suvs?|vans?|bus(?:es)?|coaches?|vehicles?|motorcycles?|pedestrians?|track\s*(?:id\s*)?#?\s*\d+)"
_MODIFIER = r"(?:the|a|an|this|that|those|these|selected|highlighted|visible|parked|red|blue|white|silver|gr[ae]y|black|yellow|green|dark|large|small|medium)"
_SUBJECT = rf"(?:{_MODIFIER}\s+){{0,5}}{_VEHICLE}(?:\s*\(track\s+\d+\))?(?:\s+on\s+the\s+(?:left|right))?"
_LANDMARK = r"(?:buildings?|towers?|skyscrapers?|offices?|malls?|schools?|hotels?|hospitals?|shops?|storefronts?|stores?|bridges?|overpasses?|houses?|walls?|entrances?|gates?|barriers?|signs?|trees?|kerbs?|curbs?|sidewalks?)"
_PLACE = rf"(?:(?:the|a|an|this|that|those|these|nearby|distant|background|large|small|tall|red|green|grey|gray|white|blue|roadside|office|apartment)\s+){{0,4}}{_LANDMARK}"
_DISTANCE = r"(?:(?:physically|actually|very|really|quite|right)\s+){0,2}(?:near(?:\s+to)?|close\s+to|next\s+to|beside|adjacent\s+to|far\s+from|far\s+away\s+from)"
_PREDICATE = r"(?:(?:parked|stopped|standing|sitting|located|positioned|waiting|driving)\s+)?"
_QUESTION = re.compile(rf"\b(?:is|are|was|were)\s+(?P<subject>{_SUBJECT})\s+{_PREDICATE}{_DISTANCE}\s+(?P<place>{_PLACE})\b",re.I)
_INDIRECT = re.compile(rf"\b(?:if|whether)\s+(?P<subject>{_SUBJECT})\s+(?:is|was)\s+{_PREDICATE}{_DISTANCE}\s+(?P<place>{_PLACE})\b",re.I)
_HOW_CLOSE = re.compile(rf"\bhow\s+(?:close|far(?: away)?)\s+(?:is|was|are|were)\s+(?P<subject>{_SUBJECT})\s+(?:to|from)\s+(?P<place>{_PLACE})\b",re.I)
_IMAGE_SCOPE = re.compile(r"\b(?:image[- ]plane|pixels?|on[- ]screen|in (?:the|this) (?:2d|two[- ]dimensional) (?:image|view))\b|"
                          r"^\s*(?:in|within) (?:the|this) (?:image|picture|frame)\s*[,;]\s*|"
                          r"\b(?:appear|look|seem)\s+(?:visually\s+)?(?:near|close|next to|beside)\b.{0,100}\b(?:image|picture|screen|frame|view)\b",re.I)
_PHYSICAL_CLAIM = re.compile(rf"\b{_DISTANCE}\s+(?:{_PLACE}|(?:each|one)\s+(?:other|another)|it|them|its\s+entrance)\b|"
                             r"\b(?:physically (?:near|close|adjacent|far|separated)|(?:close|far) (?:together|apart) in (?:reality|real life)|"
                             r"in (?:real life|reality|the real world)|within (?:walking distance|reach)|"
                             r"(?:\d+|one|two|three|few|several) (?:steps?|blocks?|car lengths?)|"
                             r"nearby|in close proximity|metres?|meters?|kilometres?|kilometers?|feet|yards?|miles?)\b",re.I)
_IMAGE_DETAIL = re.compile(r"\b(?:left|right|above|below|upper|lower|top|bottom|cent(?:re|er)|overlap\w*|foreground|background|separat\w*|edge)\b",re.I)
_TIME_PREFIX = re.compile(r"^\s*(?:at|in|by)\s+(?:(?:frame\s*#?\s*\d+)|(?:\d+:\d{2}(?:\.\d+)?)|(?:\d+(?:\.\d+)?\s*(?:seconds?|s)))\s*,?\s*",re.I)
_LIMIT = re.compile(r"\b(?:cannot|can[’']t|can not|could not|does not|do not|is not|isn[’']t|unable to)\s+"
                    r"(?:be\s+)?(?:accurately\s+|actually\s+|reliably\s+)?(?:establish|determine|measure|infer|judge|tell|confirm|know|provide|show|known|measurable)\w*\b",re.I)


def physical_question(words):
    if _IMAGE_SCOPE.search(words) and not re.search(r"\b(?:physically|actual(?:ly)?|real[- ]world)\b",words,re.I):
        return None
    match = _QUESTION.search(words) or _HOW_CLOSE.search(words) or _INDIRECT.search(words)
    if match and re.fullmatch(r"\s*(?:(?:please[, ]+)?(?:tell me[, ]+|can you (?:tell me|check)[, ]+)?|"
                              r"in (?:the|this) (?:image|picture|frame),?\s*)",_TIME_PREFIX.sub("",words[:match.start()]),re.I):
        return match
    return None


def image_question(match):
    return (f"In this frame, describe the image-plane positions of {match['subject']} and {match['place']}. "
            "Where do they appear relative to each other in the picture?")


def can_reframe(match):
    # A visual follow-up cannot discard a second condition or a referenced earlier time.
    return bool(match and not _TIME_PREFIX.match(match.string) and not match.string[match.end():].strip(" ?!."))


def limitation(frame, track=None):
    subject = f"Track {track} is selected in frame {frame}." if track is not None else f"This is frame {frame}."
    return f"{subject} The camera and recorded tracks do not measure the physical distance to that landmark."


def unsupported_image_answer(words, answer):
    """A camera-only visual comparison must give an image observation without claiming physical separation."""
    image_scope = _IMAGE_SCOPE.search(words) or re.search(r"\boverlap\w*\b.{0,70}\b(?:picture|image|frame|screen)\b",words,re.I)
    relation = re.search(r"\b(?:positions?|near|close|beside|overlap\w*|above|below|left of|right of)\b",words,re.I)
    describes = re.search(r"\b(?:what|which) colou?r|\bhow many\b|\bcarrying\b",words,re.I)
    if not (image_scope and relation and not describes and re.search(rf"\b{_LANDMARK}\b",words,re.I)):
        return False
    clauses = re.split(r"[.;!?]|\b(?:but|however|whereas|although)\b",answer,flags=re.I)
    overclaim = any(_PHYSICAL_CLAIM.search(clause) and not _LIMIT.search(clause) for clause in clauses)
    return overclaim or not _IMAGE_DETAIL.search(answer)
