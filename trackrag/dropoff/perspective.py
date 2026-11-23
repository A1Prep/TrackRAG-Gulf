"""Clarify a passenger's forward reference using alternatives in the actual frame."""
import re

from .passenger import instruction_context, unresolved
from .referent_policy import clarify_reference as original_clarification


POLICY = "vehicle-relative-to-passenger-v1"

# A camera-relative expression supplies a viewpoint. A relationship to another
# landmark can distinguish one vehicle, so leave its complete phrase for Qwen.
_OCCUPANT = r"(?:us|me|you|(?:our|my|your) (?:own )?(?:vehicle|car|cab|taxi))"
_ADVERBS = r"(?:(?:just|right|straight|directly|immediately|up|out)\s+)*"
_FORWARD = re.compile(
    rf"\b{_ADVERBS}(?:ahead(?:\s+of\s+{_OCCUPANT}\b|(?!\s+of\b))"
    rf"|in\s+front(?:\s+of\s+{_OCCUPANT}\b|(?!\s+of\b)))\b",
    re.IGNORECASE,
)


def _viewpoint_clarification(text, ranking, frame_size):
    phrases = list(_FORWARD.finditer(text))
    if not phrases:
        return None
    normalized = _FORWARD.sub("ahead", text)
    if normalized.casefold() == text.casefold():
        return None
    choice = original_clarification(normalized, ranking, frame_size)
    if not choice or len(choice["track_ids"]) < 2:
        return None
    return {**choice, "policy": POLICY, "base_policy": choice["policy"],
            "viewpoint_phrases": [m.group() for m in phrases]}


def clarify_reference(text, ranking, frame_size):
    """Apply both policies to the same original, learned candidate ranking."""
    return (original_clarification(text, ranking, frame_size)
            or _viewpoint_clarification(text, ranking, frame_size))


class PassengerViewpoint:
    """Add one dialogue check to successful proposals from the released grounder."""

    def __init__(self, model):
        self.model = model

    def interpret(self, request):
        return self.review(request, self.model.interpret(request))

    @staticmethod
    def review(request, outcome):
        if not outcome.get("ok") or outcome["result"]["status"] != "proposed":
            return outcome
        provenance = outcome["provenance"]
        ranking = provenance.get("referent_ranking")
        if ranking is None:
            return outcome
        frame = request["scene"]["frame"]
        advice = _viewpoint_clarification(
            instruction_context(request), ranking, (frame["width"], frame["height"])
        )
        if advice is None:
            return outcome
        return {
            **outcome, "result": unresolved(advice["question"]),
            "provenance": {**provenance, "viewpoint_clarification": {
                "decision": advice, "original_proposal": outcome["result"],
            }},
        }

    def passenger_area(self, request, box):
        return self.model.passenger_area(request, box)

    @property
    def requires_verified_manual_surface(self):
        return getattr(self.model, "requires_verified_manual_surface", False)
