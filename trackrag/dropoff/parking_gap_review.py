"""Reinspect a central, unoccupied parking patch after an inconclusive full-frame reading."""
from copy import deepcopy

from .parking_reference_board import steps as reviewed_steps
from .parking_steps import _json
from .parking_surface import _contact, options_for
from .passenger import proposed
from .passenger_parking import parking_instruction


GAP_PROMPT = (
    "Classify the parking-row layout at the marked point between the two outlined vehicles. "
    "Imagine a passenger describing the location, and look at the actual spaces and vehicles in this photograph. "
    "Use 'opening' if there is a visibly empty parking space or a clear interruption in the parked row at the point, "
    "'occupied' for adjacent, overlapping or intervening vehicles with no such space, 'elsewhere' for the driving lane or pavement, "
    "or 'uncertain' if you cannot distinguish them. A painted stall outline is optional. "
    "Also check whether the two outlined vehicles are the passenger's landmarks. "
    "Return JSON with keys references_match (boolean) and layout (one of these four terms)."
)
REVIEWABLE = {
    "model_declined_parking_components", "model_declined_supported_components", "selected_parking_component_not_free",
}


def central_to_pair(option):
    """Require the extra marker to fall in the middle third between the two vehicles' ground contacts."""
    ax, ay, aw, ah = _contact(option["anchor_boxes"][0])
    bx, by, bw, bh = _contact(option["anchor_boxes"][1])
    width, height = (aw + bw) / 2, (ah + bh) / 2
    dx, dy = (bx - ax) / width, (by - ay) / height
    length_squared = dx * dx + dy * dy
    if not length_squared:
        return False
    px, py = option["point"]
    fraction = ((px - ax) / width * dx + (py - ay) / height * dy) / length_squared
    return 1 / 3 <= fraction <= 2 / 3


def confirms_opening(response):
    result = _json(response)
    layouts = ("opening", "occupied", "elsewhere", "uncertain")
    if (not isinstance(result, dict) or set(result) != {"references_match", "layout"}
            or type(result["references_match"]) is not bool or result["layout"] not in layouts):
        raise ValueError("Expected a reference match and a parking-row classification")
    return result["references_match"] and result["layout"] == "opening"


def steps(request, classes):
    previous = yield from reviewed_steps(request, classes)
    if previous["reason"] not in REVIEWABLE:
        return previous
    effective = deepcopy(request)
    effective["text"] = parking_instruction(request)
    # A fragment under one bumper or an adjacent driving lane cannot resolve the earlier uncertainty.
    options = options_for(effective, classes, previous["pair"], max_options=1)
    if not options or not central_to_pair(options[0]):
        return previous
    option = options[0]
    response = yield {"kind": "gap_inspection", "system": GAP_PROMPT,
                      "view": {"type": "inspection", "option": option}}
    try:
        confirmed = confirms_opening(response)
    except (TypeError, ValueError):
        return {"error": "invalid_visual_review", "reason": "malformed_review_reply"}
    if not confirmed:
        return previous
    return {"result": proposed(option["point"], option["region_box_xyxy"], option["anchor_boxes"]),
            "reason": "visually_confirmed_parking_opening", "pair": option["tracks"],
            "choice": option["choice"], "surface": option["surface"]}
