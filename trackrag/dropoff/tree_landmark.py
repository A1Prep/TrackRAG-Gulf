"""Locate a pictured canopy independently of whether a passenger can get out beside it."""
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import bay_intent, bay_landmarks, manual_offlane, manual_surfaces
from .parking_evidence import source_input


POLICY = "pictured_tree_and_selected_surface_v2"
CHOICE = "parking_on_pictured_tree_frontage"
ASK_SIDES = "The blue outlines show roadside greenery on both sides of the photo. Which tree did you mean?"
_SIDE_REPLY = re.compile(
    r"(?:please[, ]+)?(?:(?:the|that) (?:tree|one) )?(?:(?:on|to) (?:the |my )?)?"
    r"(?:the )?(left|right)(?:[- ]hand)?(?: (?:side|one|tree))?(?:[, ]+please)?[.!]?", re.I)
_SHOPS = re.compile(r"\b(?:shop|store|market|grocery|pharmacy|electronics|mobiles|restaurant|caf[eé]|cafeteria|bakery)\b", re.I)
_UNCERTAIN_PLACE = re.compile(r"\b(?:no|none|unknown|maybe|possible|possibly|unclear|unreadable)\b|\?", re.I)


@dataclass(frozen=True)
class TreeRequest:
    side: str
    singular: bool
    palm: bool = False
    size: str | None = None
    shops: bool = False


def _instruction(text):
    if (not isinstance(text, str) or not re.search(r"\b(?:trees?|palms?|vegetation|greenery)\b", text, re.I)
            or not bay_intent.permitted_destination(text)):
        return None
    words = bay_intent.parse(bay_intent.destination_words(text))
    if (not words or words.blocked or words.uncertain or words.vehicles or words.rank or words.orientation
            or not words.frontage_phrases):
        return None
    tree, *other = words.frontage_phrases
    if (not tree or tree[-1] not in {"tree", "greenery", "vegetation", "palm"}
            or tree[:-1] not in ((), ("tall",), ("big",), ("large",))):
        return None
    if other and (other != [("shop",)] or not re.search(
            r"\b(?:outside|in front of|by|beside|near)\s+(?:(?:the|those|these)\s+)?(?:shops?|stores?)\b", text, re.I)):
        return None
    return TreeRequest(words.side or "either", bool(re.search(r"\b(?:tree|palm)\b", text, re.I)),
                       palm=tree[-1] == "palm", size=tree[0] if len(tree) == 2 else None, shops=bool(other))


def requested(request):
    """A short side answer requires the immediately preceding tree question on this frame."""
    text = request.get("text", "")
    if not isinstance(text, str):
        return None
    intent = _instruction(text)
    if intent:
        return intent
    reply = _SIDE_REPLY.fullmatch(" ".join(text.split()))
    turns = request.get("dialogue", [])
    if not reply or len(turns) < 2:
        return None
    previous, question = turns[-2:]
    frame = request["scene"]["frame"]
    if (previous.get("role") != "user" or question.get("role") != "assistant"
            or question.get("text") != ASK_SIDES
            or any(t.get("frame_id") != frame["frame_id"] or t.get("recording_id") != frame["recording_id"]
                   for t in (previous, question))
            or (prior := _instruction(previous["text"])) is None or prior.side != "either"):
        return None
    return TreeRequest(reply[1].lower(), prior.singular, prior.palm, prior.size, prior.shops)


def _canopies(classes, frame, side):
    import cv2
    height, width = classes.shape
    _, labels, stats, _ = cv2.connectedComponentsWithStats((classes == 30).astype("uint8"))
    found = []
    for index, (left, top, w, h, area) in enumerate(stats[1:], 1):
        if (area < .012*height*width or h < .15*height or w < .08*width or w > .85*width
                or top > .52*height or top+h < .29*height):
            continue
        pixels = labels[top:top+h, left:left+w] == index
        # A tree can be close to the centre line and still lie almost entirely on one side.
        right = np.mean(left + np.nonzero(pixels)[1] + .5 > width/2)
        edge = "right" if right >= .8 else "left" if right <= .2 else None
        if edge is None or side not in ("either", edge):
            continue
        low, high = np.quantile(np.argmax(pixels, axis=0), [.1, .9])
        crown = (low + .12*h < high)
        margin = cv2.dilate(pixels.astype("uint8"), np.ones((3,3), "uint8")).astype(bool) & ~pixels
        margin[int(.65*h):] = False
        sky = classes[top:top+h, left:left+w][margin]
        sky_backed = len(sky) > 0 and np.mean(sky == 27) >= .15
        widths = np.count_nonzero(pixels, axis=1)
        crown_width = np.median(widths[int(.25*h):max(int(.25*h)+1, int(.55*h))])
        lower_width = np.median(widths[int(.65*h):max(int(.65*h)+1, int(.85*h))])
        # A bounded, sky-backed crown tapering into a narrower lower section supports
        # one tree-shaped landmark. A cropped mass of vegetation or a hedge does not.
        isolated = bool(crown and sky_backed and left > 0 and left+w < width and top > 0
                        and h >= .25*height and crown_width > 0 and lower_width < .55*crown_width)
        box = [round(float(v), 2) for v in
               (left*frame["width"]/width, top*frame["height"]/height,
                (left+w)*frame["width"]/width, (top+h)*frame["height"]/height)]
        found.append((edge, box, bool(crown and sky_backed), isolated))
    return sorted(found, key=lambda item: (item[0], item[1][0]))


def _ground_message(reason, evidence, classes):
    """Describe visible ground without deciding legal access or vehicle clearance."""
    if reason == "unverified_surface" and "center_cell_xy" in evidence:
        x, y = evidence["center_cell_xy"]
        area = classes[max(0,y-2):y+3, max(0,x-2):x+3]
        center = evidence["center_class_id"]
        if center == 30 and np.mean(area == 30) >= .75:
            return "Your mark is on vegetation or a planted area. I can’t verify a place for the car there."
        if center == 15 and np.mean(area == 15) >= .75:
            return ("The surface check reads your mark as pedestrian paving. It may also serve a driveway, "
                    "but I can’t verify a stopping place at that point.")
        if center in (8, 23):
            return "Your mark touches pedestrian crossing markings."
    if reason == "through_lane":
        return "The surface check reads your mark as road. I couldn’t verify a parking spot there."
    if reason == "observed_object":
        return "Your mark touches a detected object."
    if reason == "parking":
        return "The surface at your mark looks like parking, but I can’t tie that point to the tree you described."
    return "I can’t verify an open stopping place at your mark."


def _on_same_frontage(classes, frame, point, canopy, shops):
    """Trace from actual canopy pixels down to the checked parking, requiring a sidewalk and kerb in between."""
    h, w = classes.shape
    scale = np.array([w/frame["width"], h/frame["height"]])
    cx, cy = np.floor(np.asarray(point)*scale).astype(int)
    x0, y0, x1, y1 = np.asarray(canopy)*np.tile(scale, 2)
    if not (0 <= cx < w and x0 <= cx <= x1 and y0 < cy < h):
        return False
    radius = max(2, round(.006*w))
    columns = range(max(0, cx-radius), min(w, cx+radius+1))
    corresponding = 0
    for x in columns:
        green = np.flatnonzero(classes[int(y0):min(cy,round(y1)+1),x] == 30)
        if not len(green):
            continue
        profile = classes[int(y0)+green[-1]+1:cy+1,x]
        sidewalk = np.flatnonzero(profile == 15)
        curb = np.flatnonzero(profile == 2)
        if (not len(sidewalk) or not len(curb) or sidewalk[-1] > curb[-1]
                or np.any(np.isin(profile, (3,4,5,6,8,13,23,19,20,21,22,52,54,55,57,61)))
                or (shops and not np.any(profile[:sidewalk[-1]] == 17))):
            continue
        corresponding += 1
    return corresponding >= .75*len(columns)


def _parking(request, photo, surface, classes, canopy, shops, review):
    from . import bay_clearance, surface_stalls, whole_bay
    scene = request["scene"]
    frame = scene["frame"]
    road = bay_clearance.visible_road_component(classes)
    if road is None:
        return []
    _, _, _, options = whole_bay._layout(request, photo, surface, review)
    matched = []
    for option in options:
        point = option["point"]
        if not _on_same_frontage(classes, frame, point, canopy, shops):
            continue
        mark = [point[0]-2,point[1]-2,point[0]+2,point[1]+2]
        reason, _ = manual_offlane.surface_check(frame, scene["tracks"], mark, point, classes)
        if (reason != "parking" or not bay_clearance.vehicle_sized(option, scene)
                or not bay_clearance.on_visible_road(option, road, frame)):
            continue
        if option["evidence"]["recipe"] == surface_stalls.RECIPE:
            # The checked perimeter can be just a piece of an open stall. Offer its pictured point only.
            option["evidence"] = {"recipe": POLICY, "coverage": "interior_point_only"}
        matched.append(option)
    return matched


def build(request, photo, surface, *, box=None, landmarks=None, stall_review=None):
    import cv2
    from . import bay_clearance, whole_bay
    intent = requested(request)
    if not intent:
        return None
    side = intent.side
    clean, sha, portable = source_input(request, photo)
    scene = clean["scene"]
    if surface is None:
        extra = {}
        if box is not None:
            _, _, mark, center = manual_offlane.mark_input(clean, box)
            extra.update(selected_box=mark, mark_center_xy=center)
        return whole_bay._make(
            clean, portable, [], None, method=whole_bay.AUTO if box is None else whole_bay.MARK,
            reason="tree_surface_evidence_unavailable", tree_reference_policy=POLICY,
            clearance_policy=whole_bay.CLEARANCE_POLICY, operational_status="unavailable",
            question="I don’t have a verified tree and ground check for this photo. Could you name a different landmark?",
            **extra)
    classes = manual_surfaces.decode(surface, scene["frame"], sha)
    canopies = _canopies(classes, scene["frame"], side)
    width, height = scene["frame"]["width"], scene["frame"]["height"]
    if intent.size:
        canopies = [row for row in canopies if row[1][3]-row[1][1] >= .3*height
                    and (intent.size == "tall" or row[1][2]-row[1][0] >= .15*width)]
    sides = {row[0] for row in canopies if row[2]}
    selector = side == "either" and sides == {"left", "right"}
    identified = side if side != "either" and intent.singular and not intent.palm and len(canopies) == 1 and canopies[0][3] else None
    labels = bay_landmarks.verify(landmarks, photo, (width, height)) if landmarks else None
    phrases = labels[0 if side == "left" else 2].split("|") if labels and side != "either" else []
    shops = any(_SHOPS.search(phrase) and not _UNCERTAIN_PLACE.search(phrase) for phrase in phrases)
    if intent.shops and not shops:
        identified = None
    if identified:
        description = f"The blue outline shows the tree-shaped canopy on the {side}."
    elif canopies and intent.palm:
        description = "The blue outline shows tall roadside vegetation" + (f" on the {side}" if side != "either" else "") + ". I can’t verify that it is a palm."
    elif canopies and intent.shops and shops:
        description = f"The photo shows a canopy above the shops on the {side}. I can’t pinpoint its trunk."
    elif canopies:
        where = "on both sides" if len({c[0] for c in canopies}) == 2 else f"on the {canopies[0][0]}"
        description = f"I can outline roadside greenery {where}, but can’t distinguish an individual tree with confidence."
    else:
        where = " on the " + side if side != "either" else ""
        description = f"I can’t distinguish the tree{where} in this photo."
    if intent.shops and not shops:
        description += " I couldn’t verify the shops you mentioned there."
    extra = {}
    if landmarks:
        extra["landmark_evidence"] = landmarks
    parking = []
    if (side != "either" and len(canopies) == 1 and not intent.palm
            and (not intent.shops or shops) and np.count_nonzero(classes == 10) >= .0004*classes.size
            and bay_clearance.visible_road_component(classes) is not None):
        parking = _parking(clean, photo, surface, classes, canopies[0][1], intent.shops, stall_review)
        if stall_review:
            extra["stall_review"] = stall_review
    tap = None
    if box is not None:
        _, _, mark, center = manual_offlane.mark_input(clean, box)
        reason, measured = manual_offlane.surface_check(scene["frame"], scene["tracks"], mark, center, classes)
        corners = [(x,y) for x in (mark[0],mark[2]) for y in (mark[1],mark[3])]
        matches = [option for option in parking if reason == "parking" and all(
            cv2.pointPolygonTest(np.float32(option["polygon"]), corner, False) >= 0 for corner in corners)]
        if parking and not matches and _instruction(clean["text"]):
            # The button rechecks this stand-alone wording only on passenger request.
            # The rejected tap itself receives neither a proposal nor a selectable pin.
            extra["tree_parking_available"] = True
        parking = []
        if len(matches) == 1:
            # An explicit point is grounded only when its entire mark lies inside the very
            # parking region independently associated with the requested canopy.
            parking = [{"point":center,"box":mark,"polygon":[list(corners[i]) for i in (0,2,3,1)],
                        "evidence":{"recipe":POLICY,"coverage":"interior_point_only"}}]
            tap = center
        question = f"{description} {_ground_message(reason, measured, classes)} You can mark a different off-lane spot."
        extra.update(selected_box=mark, mark_center_xy=center, tree_mark_surface=measured)
    elif selector:
        question = ASK_SIDES
    elif parking:
        question = description + " The highlighted parking is on the same pictured frontage. Is this where you meant to get out?"
    else:
        question = description + " I couldn’t verify an open pull-in beside it. You can mark an off-lane spot."
    outcome = whole_bay._make(
        clean, portable, parking, surface, method=whole_bay.AUTO if box is None else whole_bay.MARK,
        reason=CHOICE if parking else "vegetation_seen_space_withheld" if canopies else "tree_extent_unverified",
        question=question, tap=tap, require_choice=box is None, tree_reference_policy=POLICY,
        clearance_policy=whole_bay.CLEARANCE_POLICY, **extra)
    if canopies:
        outcome["result"]["anchor_boxes_xyxy"] = [row[1] for row in canopies]
    if selector and box is None:
        outcome["provenance"]["tree_side_choices"] = ["left", "right"]
    if identified:
        outcome["provenance"]["tree_landmark_side"] = identified
    return outcome


def available(request, *, box=None, allow_lookup=True):
    from . import surface_stalls
    intent = requested(request)
    if not intent:
        return None
    sha = request["scene"]["provenance"]["image_sha256"]
    photo = Path(request["scene"]["frame"]["image_path"]).read_bytes()
    surface = manual_surfaces.get_packet(sha) if allow_lookup else None
    landmarks = bay_landmarks.get_packet(sha) if surface and intent.shops else None
    review = surface_stalls.get_packet(sha) if surface and intent.side != "either" and not intent.palm else None
    return build(request, photo, surface, box=box, landmarks=landmarks, stall_review=review)


def require_current_ground(request, outcome):
    """A new offer or Save needs a measured link to this tree; archived decisions retain their original verifier."""
    if not requested(request):
        return
    source = (outcome or {}).get("provenance", {})
    if source.get("method") == "passenger_selected_painted_bay_v1":
        source = source.get("choice_source", {}).get("provenance", {})
    from . import whole_bay
    if (source.get("method") not in {whole_bay.AUTO, whole_bay.MARK}
            or source.get("tree_reference_policy") != POLICY or source.get("reason") != CHOICE):
        raise ValueError("Recheck the tree and the parking beside it before saving this location.")
