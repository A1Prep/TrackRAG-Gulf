"""Ask about the full set of parking components, then validate the chosen original pixel."""
from copy import deepcopy

import numpy as np

from tools.passenger_parking_anchors import resolve_anchor_reply
from tools.passenger_parking_candidates import options_for as parking_components
from tools.passenger_parking_study import parse_choice
from tools.passenger_road_points import MARKED
from .parking_steps import WHERE, _ask, steps as guarded_steps
from .parking_surface import _contact, _geometry, _patch
from .passenger import proposed
from .passenger_parking import parking_instruction


# Vertical Mapillary obstacles whose support can occupy the parking row.
STRUCTURES = (3, 4, 5, 6, 16, 17, 18, 33, 34, 38, 39, 40, 42, 44, 45, 47)


def structural_obstruction(classes, geometry):
    """Look for a solid structure extending to the estimated contact line within the gap.

    A distant wall above the vehicle contact line and an isolated semantic pixel
    provide insufficient evidence to veto a candidate. Overlapping boxes provide
    no exposed horizontal interval for this test.
    """
    if geometry is None:
        return False
    left, right = sorted(geometry["boxes"], key=lambda box: box[0])
    start, stop = int(np.ceil(left[2])), int(np.floor(right[0]))
    if stop - start < 5:
        return False
    ax, ay, aw, ah = _contact(left)
    bx, by, bw, bh = _contact(right)
    if bx <= ax:
        return False
    x = np.arange(max(0, start), min(classes.shape[1], stop))
    if not x.size:
        return False
    along = (x + .5 - ax) / (bx - ax)
    ground = ay + along * (by - ay)
    height = ah + along * (bh - ah)
    # Require vertical continuity across the expected foot of the landmarks.
    y = np.rint(ground[None, :] + height[None, :] * np.linspace(-.25, .025, 7)[:, None]).astype(int)
    inside = ((y >= 0) & (y < classes.shape[0])).all(axis=0)
    solid = np.isin(classes[y.clip(0, classes.shape[0] - 1), x], STRUCTURES).all(axis=0) & inside
    edges = np.flatnonzero(np.diff(np.r_[False, solid, False].astype(int))).reshape(-1, 2)
    width = min(aw, bw)
    return any(end - begin >= max(4, .07 * width) for begin, end in edges)


def selected_region(classes, geometry, point):
    if geometry is None:
        return None
    sx, sy = geometry["scale"]
    x, y = int(point[0] / sx), int(point[1] / sy)
    if not (0 <= x < classes.shape[1] and 0 <= y < classes.shape[0]):
        return None
    supported = (classes == 10) & geometry["free"] & geometry["parcel"]
    radius = min(18, max(2, round(.1 * geometry["local_h"][y, x])))
    cells = _patch(supported, x, y, radius)
    if cells is None:
        return None
    left, top, right, bottom = cells
    return [left * sx, top * sy, right * sx, bottom * sy]


def choose(board, kind):
    letters = [option["choice"] for option in board]
    reply = yield {"kind": kind, "system": MARKED + " Offered letters: " + ", ".join(letters),
                   "view": {"type": "parking_components", "options": board}}
    return parse_choice(reply, letters)["choice"]


def steps(request, classes):
    """Preserve the established letter board and the passenger's right to decline."""
    fallback = guarded_steps(request, classes)
    try:
        first = next(fallback)
    except StopIteration as done:
        return done.value
    answer = yield first  # Same first reading and original image in both recipes.
    effective = deepcopy(request)
    effective["text"] = parking_instruction(request)
    anchor = resolve_anchor_reply(effective, answer)
    pair = anchor["track_ids"] if anchor["state"] == "matched" else []
    board = parking_components(effective, classes, "parking", pair) if pair else []
    if board:
        geometry = _geometry(classes, effective, pair)
        if structural_obstruction(classes, geometry):
            return _ask("structural_obstruction_in_requested_row", WHERE, pair)
        try:
            letter = yield from choose(board, "parking_components")
        except (TypeError, ValueError):
            return {"error": "invalid_location_selection", "reason": "malformed_choice_reply"}
        if letter is None:
            return _ask("model_declined_parking_components", WHERE, pair)
        choice = next(o for o in board if o["choice"] == letter)
        region = selected_region(classes, geometry, choice["point"])
        if region is None:
            regions = {o["choice"]: selected_region(classes, geometry, o["point"]) for o in board if o != choice}
            remaining = [o for o in board if regions.get(o["choice"]) is not None]
            if not remaining:
                return _ask("selected_parking_component_not_free", WHERE, pair)
            try:
                letter = yield from choose(remaining, "supported_components")
            except (TypeError, ValueError):
                return {"error": "invalid_location_selection", "reason": "malformed_choice_reply"}
            if letter is None:
                return _ask("model_declined_supported_components", WHERE, pair)
            choice = next(o for o in remaining if o["choice"] == letter)
            region = regions[letter]
        return {"result": proposed(choice["point"], region, choice["anchor_boxes"]), "pair": choice["tracks"], "choice": letter,
                "reason": "visible_roadside_proposal", "surface": "parking"}
    # Only where a conventional parking board doesn't exist, inspect the guarded
    # roadside alternatives without repeating the first GPU read.
    try:
        step = fallback.send(answer)
        while True:
            response = yield step
            step = fallback.send(response)
    except StopIteration as done:
        result = done.value
    if result.get("result", {}).get("status") == "proposed":
        pair = result["pair"]
        if structural_obstruction(classes, _geometry(classes, effective, pair)):
            return _ask("structural_obstruction_in_requested_row", WHERE, pair)
    return result
