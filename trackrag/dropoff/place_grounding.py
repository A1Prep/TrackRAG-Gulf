"""Ground place and bay requests on visible parking pixels from the current source photograph."""
from hashlib import sha256
from io import BytesIO
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tools.passenger_place_call import PROCESSOR_REVISION as PROCESSOR

from .passenger import instruction_context, proposed, unresolved

METHOD = "trained_qwen_place_with_parking_surface_v3"
BASE = "10d842936325905934ea24e64325d774693f166e"
ADAPTER = "76bfb3230b17a86248b52ca6a9f8d2ffca324aa80216d6b93845ccb12ad86743"
PLACE = re.compile(
    r"\b(?:bays?|parking (?:spots?|spaces?|stalls?)|(?:marked|painted|empty) (?:spaces?|stalls?)|"
    r"shopfronts?|storefronts?|shops?|stores?|caf[eé]s?|restaurants?|pharmac(?:y|ies)|hotels?|malls?|supermarkets?|"
    r"garages?|entrances?|doorways?|driveways?|forecourts?|villas?|gates?|buildings?|receptions?|lobb(?:y|ies)|kiosks?|"
    r"mosques?|schools?|clinics?|hospitals?|offices?|terminals?|petrol (?:station|pump)s?|"
    r"trees?|palms?|bush(?:es)?|hedges?|shrubs?|flower[ -]?beds?|planters?|bench(?:es)?|fountains?|"
    r"signposts?|lamp[ -]?posts?|streetlights?|bollards?)\b", re.I)
VEHICLE = re.compile(r"\b(?:cars?|vehicles?|suvs?|sedans?|taxis?|trucks?|lorr(?:y|ies)|buses|bus|minivans?|vans?|motorcycles?|"
                     r"patrol|camry|land cruiser|corolla|tesla|jeep|lexus|toyota|nissan|honda|bmw|mercedes)\b", re.I)
VEHICLE_COMPOUND = re.compile(r"\b(?:school bus|hotel (?:shuttle|bus|van)|delivery (?:van|truck))\b", re.I)
LANDMARK_PHRASE = re.compile(r"\b(?:near|beside|by|at|in front of|next to|behind|outside|opposite|"
                             r"(?:left|right) of|in)\s+(?:the|that|those|this|a|an)\s+", re.I)
# A business name may have no article, as in "at Al Noor". Preserve case for the name.
NAMED_BUSINESS = re.compile(r"\b(?i:near|beside|by|at|outside|opposite|in front of|next to)\s+"
                            r"([A-Z][\w’'-]+(?:[ \t]+[A-Z][\w’'-]+){0,4})\b")
CLOCK_OR_GESTURE = frozenset("midnight noon sunrise sunset dawn dusk monday tuesday wednesday thursday friday "
                             "saturday sunday am pm here there it this that those them the same left right one".split())
UNSPECIFIC = frozenset("one ones side other nearer closer nearest farthest far right left here there middle front "
                       "back ahead same next dark light red white black blue green spot part open empty of the on "
                       "at to this that and please just me us you can could would stop get out".split())
NON_LANDMARK_TERMS = UNSPECIFIC | CLOCK_OR_GESTURE | frozenset(
    "a an it them previous usual place point around exactly about today tonight tomorrow yesterday later "
    "morning afternoon evening night time lunch lunchtime after before in hour hours minute minutes "
    "one two three four five six seven eight nine ten eleven twelve fifteen twenty thirty forty fifty half past quarter".split())


def has_landmark_terms(phrase):
    """A clock reading or gesture alone cannot identify an object in the photograph."""
    # Keep the noun in 'at the clock', while treating 'five o'clock' and '5 p.m.' as times.
    phrase = re.sub(r"\bo\s*['’]?\s*clock\b|\b[ap]\.\s*m\b", " ", phrase.casefold())
    return any(word not in NON_LANDMARK_TERMS for word in re.findall(r"[^\W\d_]+", phrase))


def named_reference(words):
    fixed = PLACE.search(words)
    relation = LANDMARK_PHRASE.search(words)
    vehicle = VEHICLE.search(words)
    if VEHICLE_COMPOUND.search(words) or (vehicle and re.search(r"\bbetween\b", words, re.I)):
        return False
    if fixed and not VEHICLE.search(words[:fixed.start()]):
        return True
    if not relation:
        business = NAMED_BUSINESS.search(words)
        return bool(business and not vehicle and has_landmark_terms(business[1]))
    # Let the image model interpret unfamiliar place nouns. Relationship words and
    # adjectives alone cannot convert an unpointed gesture into a visual target.
    scope = re.split(r"[,;.!?]|\b(?:just behind|just before|beside|next to)\b",words[relation.end():],maxsplit=1,flags=re.I)[0]
    if vehicle and (vehicle.start() < relation.end() or VEHICLE.search(scope)):
        return False
    return has_landmark_terms(scope)
CHOOSE = (
    "The numbered dots indicate the centres of different pieces of visible parking pavement. "
    "Read the passenger's words. Identify the named place or exact marked bay in the actual scene. "
    "Which dot is on the requested vacant ground and reachable from the passenger's side of the street? "
    "Look at distance, near versus far, painted stalls, raised islands, walls, parked vehicles, and access aisles. "
    "A nearby parked car or a traffic lane cannot substitute for a storefront or an empty bay. "
    "Reply only with the number. Reply NONE if no dot clearly matches or the description is ambiguous."
)
CHECK = (
    "The yellow ring marks only its centre. Independently inspect the named place, the exact point, "
    "and the route from the camera side. Is the point in the particular unoccupied stopping area the "
    "passenger describes? Check for an intervening divider, wall, pavement, vehicle, a different row "
    "or an access aisle instead of a marked bay. If another vehicle already occupies this painted stall, "
    "a strip of pavement behind its bumper is still an occupied stall. Only answer MATCH if the view supports the named "
    "reference, the location and its access. Otherwise answer BLOCKED, WRONG or UNCLEAR."
)
QUESTION = "I can’t place that request on a clear bay in this view. Could you mark the spot you mean, or name another visible landmark?"


def place_reference(request):
    """Route by the passenger's referent, leaving vehicle-first and two-object relations to their own grounder."""
    current = request["text"]
    if named_reference(current):
        words = current
    else:
        # A clarification such as "the one on the right" only needs the active passenger place instruction.
        if VEHICLE.search(current):
            return None
        words = instruction_context(request)
        if words == current:
            return None
        user_turns = [t["text"] for t in request["dialogue"] if t["role"] == "user"]
        previous = next((i for i in reversed(range(len(user_turns))) if named_reference(user_turns[i])), None)
        if previous is None:
            return None
        words = " ".join([*user_turns[previous:],current])
        if any(VEHICLE.search(t) and not named_reference(t) for t in user_turns[previous+1:]):
            return None
    if len(words) > 1600:
        return None
    if re.search(r"\bbetween\b", words, re.I) and VEHICLE.search(words):
        return None
    return words


def current_context(request):
    frame = request["scene"]["frame"]
    return {"words":place_reference(request), "size":[frame["width"], frame["height"]],
            "image_sha256":request["scene"]["provenance"]["image_sha256"],
            "current_boxes":[t["bbox_xyxy"] for t in request["scene"]["tracks"] if t["frame_id"] == frame["frame_id"]]}


def surface_options(classes, size, boxes):
    """Offer separated interior points only on predicted parking pixels, outside current detections."""
    from scipy.ndimage import distance_transform_edt

    height, width = classes.shape
    allowed = classes == 10
    for left, top, right, bottom in boxes:
        # A segmenter can label the exposed strip behind a parked vehicle as parking.
        # Withhold this narrow image-space strip as well as the detected silhouette.
        # This is a conservative image heuristic, not a physical clearance estimate.
        margin = (right-left)*.08
        x0, x1 = np.array([left-margin, right+margin]) * width / size[0]
        y0, y1 = np.array([top, bottom+(bottom-top)*.55]) * height / size[1]
        allowed[max(0, int(np.floor(y0))):int(np.ceil(y1)), max(0, int(np.floor(x0))):int(np.ceil(x1))] = False
    # A one-pixel fringe is weak support. Require a small interior neighbourhood even for a distant bay.
    distance = distance_transform_edt(np.pad(allowed, 1))[1:-1, 1:-1]
    work = distance.copy()
    yy, xx = np.ogrid[:height, :width]
    points = []
    nearby = np.asarray(boxes,dtype=float).reshape((-1,4)) * [width/size[0],height/size[1],width/size[0],height/size[1]]
    for _ in range(60):
        cy, cx = np.unravel_index(work.argmax(), work.shape)
        clearance = float(work[cy, cx])
        if clearance < max(2.5, width * .004):
            break
        # The displayed point and small extent both stay inside this patch. Never relocate a rejected point.
        px, py = (cx + .5) * size[0]/width, (cy + .5) * size[1]/height
        radius = min(clearance * .45, width * .01)
        dx, dy = radius * size[0]/width, radius * size[1]/height
        spread = max(width * .056, clearance * 1.65)
        work[((xx-cx)/spread)**2 + ((yy-cy)/max(height*.048, spread*.64))**2 <= 1.] = 0
        if len(nearby):
            bw, bh = nearby[:,2]-nearby[:,0], nearby[:,3]-nearby[:,1]
            horizontal = np.maximum(np.maximum(nearby[:,0]-cx,cx-nearby[:,2]),0)
            distance_to_box = (horizontal / bw)**2 + ((nearby[:,3]-cy) / bh)**2
            closest = int(distance_to_box.argmin())
            # Near a road user, exclude slivers at the back of an occupied row.
            # Require an open diameter of at least one third of its shorter visible dimension.
            if distance_to_box[closest] <= 4 and clearance < min(bw[closest],bh[closest])/6:
                continue
        points.append({"point":[float(px),float(py)], "box":[float(px-dx),float(py-dy),float(px+dx),float(py+dy)]})
        if len(points)==12:
            break
    return sorted(points, key=lambda p:(p["point"][0],p["point"][1]))


def frame_view(original, options, selected=None):
    image = original.convert("RGB")
    sw, sh = image.size
    image.thumbnail((1050,740), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=18)
    for i, option in enumerate(options, 1):
        if selected is not None and i != selected:
            continue
        x, y = (round(option["point"][0]*image.width/sw), round(option["point"][1]*image.height/sh))
        if selected is None:
            label = str(i)
            lx = min(max(13, x), image.width-14)
            ly = max(14, y-20) if y > 30 else min(image.height-14, y+20)
            draw.line((lx, ly, x, y), fill="black", width=2)
            draw.rectangle((lx-12,ly-11,lx+12,ly+9),fill="#102d37")
            draw.text((lx,ly),label,font=font,fill="white",anchor="mm")
        for radius, colour in ((8,"black"),(6,"#ffdc33")):
            draw.ellipse((x-radius,y-radius,x+radius,y+radius),outline=colour,width=2)
        draw.point((x,y),fill="#ffdc33")
    return image


def view_bytes(original, options, selected=None):
    out = BytesIO()
    frame_view(original, options, selected).save(out, format="JPEG", quality=92, subsampling=0)
    return out.getvalue()


def steps(context, classes, original):
    options = surface_options(classes, context["size"], context["current_boxes"])
    if not options:
        return unresolved(QUESTION), "no_visible_parking_surface"
    raw = yield "choose", CHOOSE, view_bytes(original, options)
    if raw == "NONE":
        return unresolved(QUESTION), "no_matching_place"
    if not isinstance(raw,str) or not re.fullmatch(r"[1-9]\d?", raw) or not 1 <= int(raw) <= len(options):
        raise ValueError("Invalid place candidate returned by the image model")
    choice = int(raw)
    verdict = yield "check", CHECK, view_bytes(original, options, choice)
    if verdict not in {"MATCH","WRONG","BLOCKED","UNCLEAR"}:
        raise ValueError("Invalid place verification response")
    if verdict != "MATCH":
        return unresolved(QUESTION), "location_"+verdict.lower()
    candidate = options[choice-1]
    return proposed(candidate["point"],candidate["box"]), "visible_parking_point"


def apply_steps(context, classes, original, answer):
    machine = steps(context, classes, original)
    try:
        step = next(machine)
        while True:
            kind, guide, pixels = step
            reply = answer(kind, guide, pixels)
            step = machine.send(reply)
    except StopIteration as done:
        return done.value


def call_identity(kind, guide, words, pixels):
    from .passenger import digest
    return {"stage":kind, "prompt_sha256":digest([guide,words]), "rendered_sha256":sha256(pixels).hexdigest()}
