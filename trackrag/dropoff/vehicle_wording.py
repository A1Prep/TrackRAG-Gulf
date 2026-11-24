"""Resolve a plausible vehicle word only after the passenger explicitly confirms it."""
from dataclasses import dataclass
import re

from . import bay_intent, single_vehicle


_COLOUR = "|".join(sorted(single_vehicle.COLORS | {"grey"}))
_BODY = re.compile(
    rf"\b(?P<relation>behind|beside|alongside|next\s+to)\s+"
    rf"(?P<subject>(?:(?:the|that|this|a)\s+)?(?:(?:parked|{_COLOUR})\s+)+(?P<word>card))"
    rf"(?P<side>\s+(?:on|to)\s+(?:the|our|your|my)\s+(?:left|right)(?:\s+side)?)?"
    rf"(?:\s*,?\s*please)?\s*[.!?]*\s*$", re.IGNORECASE,
)
_ACT = re.compile(r"\b(?:drop|stop|park|pull|leave|let|get)\b", re.I)
_PREFACE = set("could can would will may you please kindly i we would like to me us drop off stop park "
               "pull in over leave let get out right just".split())
_AFFIRM = {"yes","yeah","yep","yes please","yes that's right","correct","that's right","that is right","exactly",
           "the car","car","i mean the car","i meant the car"}
_NEGATIVE = re.compile(r"^(?:no|nope|not the car|i (?:didn't|did not) mean (?:the |a )?car)\b")
_LITERAL = re.compile(r"^(?:(?:no|nope)\s+)?(?:(?:i mean|i meant|it's|it is)\s+)?(?:the |a |that )?"
                      r"(?P<name>(?:(?:credit|payment) )?card (?:reader|shop|machine|terminal|kiosk))$")
_VEHICLE = r"(?:car|sedan|suv|van|truck|pickup|bus|hatchback|coupe|crossover|taxi|lorry)"
_REPLACEMENT = re.compile(
    rf"^(?:(?:no|yes|actually)\s+)?(?:(?:i (?:meant|mean)|make that)\s+)?"
    rf"(?P<subject>(?:(?:the|that|this|a)\s+)?(?:{_COLOUR})\s+{_VEHICLE})$"
)
_NO_PREFIX = re.compile(r"^(?:no|nope)\b[\s,.:;!–—-]*(?:(?:i (?:mean|meant)|make that)\b[\s,:]*)?", re.I)
_PLAIN_DECLINE = re.compile(r"^(?:no|nope)(?: (?:thanks|thank you|not (?:the|a) car|that's not (?:it|right)))?$"
                            r"|^(?:not the car|i (?:didn't|did not) mean (?:the |a )?car)$")
_VEHICLE_BY_PLACE = re.compile(
    rf"(?:(?P<placement>behind|beside|alongside|next to|in front of)\s+)?"
    rf"(?:(?:the|that|this|a)\s+)?(?:(?:{_COLOUR}|parked)\s+)?{_VEHICLE}\s+"
    r"(?P<relative>outside|beside|next to|near|by|in front of)\s+(?:(?:the|a|that)\s+)?"
    r"(?P<place>[a-z]+(?:[ -][a-z]+){0,4})", re.I,
)


@dataclass(frozen=True)
class Suggestion:
    corrected: str
    question: str
    word_span: tuple[int, int]
    subject_span: tuple[int, int]


def suggestion(words):
    if not isinstance(words, str) or not bay_intent.permitted_destination(words):
        return None
    match = _BODY.search(words)
    if not match:
        return None
    prefix = words[:match.start()]
    if not _ACT.search(prefix) or set(re.findall(r"[a-z]+",prefix.lower())) - _PREFACE:
        return None
    subject = match["subject"]
    modifiers = [word for word in subject.casefold().split() if word in single_vehicle.COLORS | {"grey","parked"}]
    if len(modifiers) != len(set(modifiers)) or len(set(modifiers) - {"parked"}) > 1:
        return None
    repaired = subject[:-4] + ("CAR" if match["word"].isupper() else "car")
    start, end = match.span("subject")
    corrected = words[:start]+repaired+words[end:]
    relation = " ".join(match["relation"].casefold().split())
    named = " ".join((repaired+(match["side"] or "")).split())
    question = (f"Your request says “{subject}”. Did you mean “{relation} {named}”? "
                "Reply yes to check that car, or tell me what you meant.")
    return Suggestion(corrected, question, match.span("word"), (start, end))


def pending(turns, frame):
    """Find the current unanswered question, including a repeated question after a photo tap."""
    from .manual_mark import MARK_PROMPT

    if not turns or turns[-1].get("role") != "assistant":
        return None
    current = (frame.get("recording_id"), frame.get("frame_id"))
    prompts = []
    for turn in reversed(turns):
        if (turn.get("recording_id"), turn.get("frame_id")) != current:
            return None
        if turn.get("role") == "user":
            original = turn.get("text")
            proposed = suggestion(original)
            if (proposed and prompts[0] == proposed.question
                    and all(text in (proposed.question, MARK_PROMPT) for text in prompts)):
                return original, proposed
            return None
        if turn.get("role") != "assistant":
            return None
        prompts.append(turn.get("text"))
    return None


def reply(request):
    """Use only the immediately preceding verified wording exchange on this exact photo."""
    prior = pending(request.get("dialogue") or [], request.get("scene", {}).get("frame", {}))
    if prior is None:
        return None
    original, proposal = prior
    raw = request["text"].strip().lower().replace("’", "'")
    if not re.fullmatch(r"[a-z\s,.'!-]+", raw):
        if _NO_PREFIX.match(raw):
            supplied = _NO_PREFIX.sub("", request["text"].strip()).strip(" \t.!?,")
            if supplied:
                return "replacement_unclear", supplied
        if not _ACT.search(raw) and re.match(r"(?:yes|yeah|yep|no|correct|right|the|that|car|is)\b", raw):
            return "unclear", None
        return None
    answer = " ".join(re.findall(r"[a-z]+(?:'[a-z]+)?", raw))
    if answer in _AFFIRM or re.fullmatch(r"(?:yes|yeah|yep)(?: i (?:mean|meant))? (?:the |a )?car",answer):
        return "accepted", proposal.corrected
    replacement = _REPLACEMENT.fullmatch(answer)
    if replacement:
        start, end = proposal.subject_span
        return "accepted", original[:start]+replacement["subject"]+original[end:]
    literal = _LITERAL.fullmatch(answer)
    if literal:
        return "declined", literal["name"]
    if _PLAIN_DECLINE.fullmatch(answer):
        return "declined", None
    if _NEGATIVE.match(answer):
        from . import vehicle_reference_qualifiers

        supplied = _NO_PREFIX.sub("", request["text"].strip()).strip(" \t.!?,")
        described = _VEHICLE_BY_PLACE.fullmatch(supplied)
        if described:
            original_match = _BODY.search(original)
            if described["placement"]:
                updated = original[:original_match.start()] + supplied
            elif not original_match["side"]:
                updated = original[:proposal.subject_span[0]] + supplied
            else:
                return "replacement_unclear", supplied
            ref = vehicle_reference_qualifiers.parse({"text":updated})
            if ref and any(q["field"] == "place" and q["value"].casefold() == described["place"].casefold()
                           and q["relation"] == described["relative"].casefold() for q in ref["qualifiers"]):
                return "accepted", updated
        return "replacement_unclear", supplied
    return None
