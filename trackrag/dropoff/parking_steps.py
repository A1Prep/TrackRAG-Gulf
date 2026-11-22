"""Live and offline replay of the decisions between the image model and the passenger."""
from copy import deepcopy
from math import hypot

from tools.passenger_parking_anchors import ANCHOR_PROMPT, resolve_anchor_reply
from tools.passenger_parking_study import _strict_json, parse_choice
from tools.passenger_road_points import MARKED
from .passenger import proposed, unresolved
from .passenger_parking import parking_instruction
from .projected_pair import MOTOR


TRACK_PROMPT = (
    "Identify the TWO vehicles described in the passenger's request. The top is the original scene with "
    "numbered identifiers, the lower tiles show those same vehicles. Use vehicle type, visible appearance, "
    "side of the street and which is nearer. Some mentioned landmarks can be present with NO opening. "
    "Identify the vehicles independently of whether there is a place to stop. Return only JSON "
    '{"ids":[first_id,second_id]} using two different displayed integer IDs, or {"ids":[]} if ambiguous.'
)
VERIFY_PROMPT = (
    "A passenger has asked for a specific roadside opening. The upper panel shows the full annotated street. "
    "The lower panel enlarges the candidate and the two vehicle outlines. The white dot is the proposed "
    "location. Check (1) both outlined vehicles fit these exact descriptions and relative positions and "
    "(2) the dot is in the visible empty roadside opening between them. The opening must be distinguishable "
    "from a through traffic lane, an occupied stall, a crossing, a driveway or pedestrian paving. A car-sized "
    "gap cannot be inferred merely because some asphalt is visible. Return only JSON "
    '{"references_match":true,"visible_opening":true} with each value true or false. Use false for uncertain.'
)
WHICH = "Which two vehicles do you mean? You can select their outlines or mark the intended road area."
WHERE = "I can’t locate a clear roadside opening between those vehicles. Can you mark the intended area?"


def visible_tracks(request):
    tracks = [t for t in request["scene"]["tracks"] if t["class_name"] in MOTOR]
    def area(t):
        x1, y1, x2, y2 = t["bbox_xyxy"]
        return (x2 - x1) * (y2 - y1)
    ordered = sorted(tracks, key=lambda t: (-area(t), t["track_id"]))[:18]
    return [{"id": t["track_id"], "box": t["bbox_xyxy"]} for t in ordered]


def _json(text):
    text = text.strip()
    if text.startswith("~~~") or text.startswith(chr(96) * 3):
        fence = text[:3]
        if text.endswith(fence):
            text = text[3:-3].removeprefix("json").strip()
    return _strict_json(text)


def _tracks_reply(text, offered):
    reply = _json(text)
    if not isinstance(reply, dict) or set(reply) != {"ids"} or not isinstance(reply["ids"], list):
        raise ValueError("Expected two vehicle IDs or an empty list")
    ids = reply["ids"]
    if ids and (len(ids) != 2 or any(type(v) is not int or v not in offered for v in ids) or ids[0] == ids[1]):
        raise ValueError("Vehicle IDs must be distinct and displayed")
    return ids


def _ask(why, question, pair=()):
    return {"result": unresolved(question), "reason": why, "pair": list(pair)}


def steps(request, classes):
    """Yield ordered model calls, then return a normal target. Identical replay consumes the original raw replies."""
    from .parking_surface import options_for
    words = parking_instruction(request)
    if not words:
        raise ValueError("The active request is outside two-vehicle parking grounding")
    effective = deepcopy(request)
    effective["text"] = words
    visible = visible_tracks(effective)
    if len(visible) < 2:
        return _ask("fewer_than_two_observed_vehicles", WHICH)
    anchor_reply = yield {"kind": "anchors", "system": ANCHOR_PROMPT, "view": {"type": "original"}}
    anchors = resolve_anchor_reply(effective, anchor_reply)
    first = anchors["track_ids"] if anchors["state"] == "matched" else []
    pair = first
    options = options_for(effective, classes, first) if first else []
    reread = False
    if not options:
        reread = True
        raw_ids = yield {"kind": "tracks", "system": TRACK_PROMPT, "view": {"type": "vehicles", "tracks": visible}}
        try:
            pair = _tracks_reply(raw_ids, {t["id"] for t in visible})
        except (ValueError, TypeError):
            return {"error": "invalid_vehicle_identification", "reason": "malformed_track_reply"}
        if not pair:
            return _ask("requested_vehicles_uncertain", WHICH)
        options = options_for(effective, classes, pair, allow_road_extension=True)
        # Road-labelled pavement needs two readings of the same vehicles as well as a separate visual check.
        if set(first) != set(pair):
            options = [o for o in options if not o["requires_visual_review"]]
    if not options:
        return _ask("no_supported_roadside_patch", WHERE, pair)
    allowed = [o["choice"] for o in options]
    raw_selection = yield {"kind":"choice", "system":MARKED + " Offered letters: " + ", ".join(allowed),
                           "view":{"type":"points", "options":options}}
    try:
        letter = parse_choice(raw_selection, allowed)["choice"]
    except (TypeError, ValueError):
        return {"error":"invalid_location_selection", "reason":"malformed_choice_reply"}
    alternate = False
    if letter is None and not reread:
        # Keep the same identified cars. An extra visual check is compulsory if
        # a different interior of the same parking surface is proposed.
        frame = effective["scene"]["frame"]
        second = options_for(effective, classes, pair, prefer_established=False)
        def different(point, other):
            return hypot((point[0]-other[0])/frame["width"],(point[1]-other[1])/frame["height"]) >= .012
        second = [o for o in second if all(different(o["point"], old["point"]) for old in options)]
        if second:
            alternate = True
            options = second
            allowed = [o["choice"] for o in options]
            reply = yield {"kind":"interior_choice", "system":MARKED + " Offered letters: " + ", ".join(allowed),
                           "view":{"type":"points","options":options}}
            try:
                letter = parse_choice(reply, allowed)["choice"]
            except (TypeError, ValueError):
                return {"error":"invalid_location_selection", "reason":"malformed_choice_reply"}
    if letter is None:
        return _ask("model_declined_offered_spots", WHERE, pair)
    chosen = next(o for o in options if o["choice"] == letter)
    if alternate or chosen["requires_visual_review"] or (reread and set(first) != set(pair)):
        raw_check = yield {"kind":"check", "system":VERIFY_PROMPT, "view":{"type":"inspection", "option":chosen}}
        try:
            check = _json(raw_check)
            if not isinstance(check, dict) or set(check) != {"references_match","visible_opening"} or any(type(v) is not bool for v in check.values()):
                raise ValueError("Missing visual check")
        except (TypeError, ValueError):
            return {"error":"invalid_visual_review", "reason":"malformed_review_reply"}
        if not all(check.values()):
            return _ask("second_visual_read_declined", WHERE if check["references_match"] else WHICH, pair)
    return {"result":proposed(chosen["point"], chosen["region_box_xyxy"], chosen["anchor_boxes"]),
            "reason":"visible_roadside_proposal", "pair":list(pair), "choice":letter, "surface":chosen["surface"]}
