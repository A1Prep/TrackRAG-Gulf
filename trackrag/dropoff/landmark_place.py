"""Localize the spoken reference, then inspect its frontage at higher effective resolution."""
from io import BytesIO
import re

from PIL import Image, ImageDraw

from .passenger import proposed, unresolved


REFERENCE = (
    "Locate the specific physical landmark or row of spaces named by the passenger in this driver-view "
    "photograph. Point at the centre of the visible landmark itself: the relevant sign, doorway, object "
    "or row, as distinct from the place where a car would stop. If it is missing or ambiguous reply NONE. "
    "Otherwise return only x,y as two integers on a 0 to 1000 scale. Origin is the top left."
)
SURFACE = (
    "The cyan ring marks the passenger's likely landmark. A vehicle may only be shown a requested spot "
    "if a distinct, visibly empty vehicle parking bay or row is present immediately by that particular "
    "landmark. One clearly empty stall is sufficient, even when other stalls have vehicles in them. Inspect the surrounding ground, occupied stalls, curb, pedestrian-only paving "
    "and lines or dividers across the route. If you see empty marked stalls beside that landmark reply "
    "EMPTY. Otherwise reply OCCUPIED, WALKWAY, TRAFFIC, DIVIDED, WRONG or UNCLEAR. Return one word."
)
VERIFY = (
    "Top: original street view. Bottom: enlarged original pixels around the yellow ring. Cyan marks "
    "the passenger's landmark and yellow marks the exact proposed ground. Is the centre of the yellow "
    "ring INSIDE a distinct vacant vehicle parking bay in the row next to this landmark? Look for the "
    "boundaries of an actual stall or row, including its curb or painted lines. A free patch behind a "
    "parked car, pedestrian pavement, an access aisle, or ground beyond posts is not an empty bay. "
    "Only mark OCCUPIED if a vehicle occupies the same stall, not if a car is visible in another stall. "
    "Return only INSIDE, OCCUPIED, WALKWAY, TRAFFIC, DIVIDED, WRONG or UNCLEAR."
)
REJECT = {"OCCUPIED", "WALKWAY", "TRAFFIC", "DIVIDED", "WRONG", "UNCLEAR"}
OBSCURED = "I couldn’t identify that landmark clearly. Could you name another landmark or mark the spot on the photo?"
NO_GROUND = "I couldn’t verify an open bay beside that landmark. Could you mark where you mean or describe another spot?"


def xy(reply, size):
    match = re.fullmatch(r"\s*\(?\s*(\d{1,4})\s*[,，]\s*(\d{1,4})\s*\)?\s*", reply) if isinstance(reply, str) else None
    if match is None or max(map(int, match.groups())) > 1000:
        return None
    return tuple(round(int(value) * (extent - 1) / 1000) for value, extent in zip(match.groups(), size))


def encoded(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=92, subsampling=0)
    return buffer.getvalue()


def marked(original, window, marks, bounds):
    image = original.crop(window).convert("RGB")
    width, height = image.size
    image.thumbnail(bounds, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    for (x, y), color in marks:
        cx = (x - window[0]) * image.width / width
        cy = (y - window[1]) * image.height / height
        for radius, line, pen in ((7, "#162426", 3), (5, color, 2)):
            draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), outline=line, width=pen)
    return image


def steps(original, options):
    w, h = original.size
    full = (0, 0, w, h)
    raw = yield "reference", REFERENCE, encoded(marked(original, full, [], (1250, 850)))
    if raw == "NONE":
        return unresolved(OBSCURED), "reference_unresolved"
    anchor = xy(raw, original.size)
    if anchor is None:
        raise ValueError("Invalid landmark coordinates returned by the place model")
    # Prefer an interior point in the cluster of parking pixels nearest the identified landmark.
    # The median limits the influence of a single isolated parking prediction at a row's edge.
    nearest = sorted(options, key=lambda item: ((item["point"][0]-anchor[0])/w)**2 +
                     ((item["point"][1]-anchor[1])/h)**2)
    close_options = [item for item in nearest[:3] if
                     abs(item["point"][0]-nearest[0]["point"][0]) <= .15*w and
                     abs(item["point"][1]-nearest[0]["point"][1]) <= .10*h]
    selection = sorted(close_options, key=lambda item: item["point"][0])[1] if len(close_options) == 3 else nearest[0]
    point = selection["point"]
    if abs(point[0]-anchor[0]) > w*.42:
        return unresolved(NO_GROUND), "landmark_far_from_bay"
    marks = [(anchor, "#25f1ff")]
    # Keep the street context on the first read. The verifier then sees an independent crop
    # from original pixels, centred on the selected bay so distant lines and vehicles enlarge.
    surface = yield "frontage_surface", SURFACE, encoded(marked(original, full, marks, (1250, 850)))
    if surface not in REJECT | {"EMPTY"}:
        raise ValueError("Invalid landmark frontage classification")
    if surface != "EMPTY":
        return unresolved(NO_GROUND), "frontage_" + surface.lower()
    marks.append((point, "#ffdc33"))
    cw, ch = max(1, round(w*.45)), max(1, round(h*.44))
    left, top = min(max(0, point[0]-cw//2), w-cw), min(max(0, point[1]-ch//2), h-ch)
    wide = marked(original, full, marks, (700, 310))
    close = marked(original, (left,top,left+cw,top+ch), marks, (700, 480))
    board = Image.new("RGB", (max(wide.width,close.width), wide.height+4+close.height), "#262626")
    board.paste(wide, ((board.width-wide.width)//2, 0))
    board.paste(close, ((board.width-close.width)//2, wide.height+4))
    verdict = yield "frontage_verify", VERIFY, encoded(board)
    if verdict not in REJECT | {"INSIDE"}:
        raise ValueError("Invalid marked-bay assessment")
    if verdict != "INSIDE":
        return unresolved(NO_GROUND), "marked_bay_" + verdict.lower()
    return proposed(point, selection["box"]), "verified_landmark_bay"
