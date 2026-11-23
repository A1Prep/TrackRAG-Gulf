"""Localize the spoken reference, then inspect its frontage at higher effective resolution."""
from io import BytesIO
import re

import numpy as np
from PIL import Image, ImageDraw

from .passenger import proposed, unresolved


REFERENCE = (
    "Locate the specific physical landmark or row of spaces named by the passenger in this driver-view "
    "photograph. Point at the centre of the visible landmark itself: the relevant sign, doorway, object "
    "or row, as distinct from the place where a car would stop. If it is missing or ambiguous reply NONE. "
    "Otherwise return only x,y as two integers on a 0 to 1000 scale. Origin is the top left."
)
GROUND = (
    "This is a closer crop of the passenger's landmark, marked by a small cyan ring. Identify the centre "
    "of an unoccupied area for a vehicle immediately beside the named landmark. Look for a marked parking "
    "space, a strip outside the traffic lane, or a vehicle entrance apron connected to this street. "
    "The dot is the landmark, not the destination. If its frontage is full, behind a barrier, across "
    "a divider, on a pedestrian area, or only a moving lane is visible, return NONE. Otherwise return "
    "only the destination point x,y as two integers from 0 to 1000 relative to THIS crop, top left origin."
)
VERIFY = (
    "Inspect a passenger's tentative drop-off on the whole photo above and a close view below. Cyan marks "
    "the requested landmark. The small yellow ring marks the exact proposed ground. Does its CENTRE fall "
    "inside an empty off-lane vehicle parking space, curb pocket or entrance driveway apron adjacent to "
    "this specific landmark? Check painted edges, bollards, parked vehicles and the connection to the "
    "camera's street. A visible sign alone is insufficient. Return BAY, APRON, LANE, SIDEWALK, OCCUPIED, "
    "DIVIDED, WRONG or UNCLEAR. Use BAY or APRON only if the marked centre actually lies inside it."
)
OBSCURED = "I can’t identify that landmark clearly. Could you name another landmark or mark the spot on the photo?"
NO_GROUND = "I found a possible landmark, but I can’t identify an empty pull-in beside it. Could you mark the area you mean or describe another spot?"


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


def steps(context, classes, original):
    w, h = original.size
    full = (0, 0, w, h)
    raw = yield "reference", REFERENCE, encoded(marked(original, full, [], (1250, 850)))
    anchor = xy(raw, original.size)
    if anchor is None:
        return unresolved(OBSCURED), "reference_unresolved"
    # Crop in the original pixels to enlarge small signs and bays on distant frontages.
    cw = max(1, round(w * .67))
    left = min(max(0, anchor[0] - cw // 2), w - cw)
    top = max(0, min(round(h * .36), anchor[1] - round(h * .15)))
    window = (left, top, left + cw, h)
    marks = [(anchor, "#25f1ff")]
    raw = yield "frontage_point", GROUND, encoded(marked(original, window, marks, (1000, 850)))
    rel = xy(raw, (cw, h-top))
    if rel is None:
        return unresolved(NO_GROUND), "frontage_unresolved"
    x, y = point = (left + rel[0], top + rel[1])
    if (y < .32*h or y > .96*h or abs(x-anchor[0]) > .39*w or
            any(l <= x <= r and t <= y <= b for l,t,r,b in context["current_boxes"])):
        return unresolved(NO_GROUND), "frontage_obstructed"
    mh, mw = classes.shape
    mx, my = round(x*(mw-1)/max(w-1,1)), round(y*(mh-1)/max(h-1,1))
    patch = classes[max(0,my-1):my+2, max(0,mx-1):mx+2]
    # A sidewalk prediction alone cannot distinguish a hotel apron. The two-view check resolves it.
    if not np.isin(patch, (7, 8, 9, 10, 13, 14, 15)).any():
        return unresolved(NO_GROUND), "frontage_without_ground"
    marks.append((point, "#ffdc33"))
    wide = marked(original, full, marks, (720, 380))
    close = marked(original, window, marks, (720, 470))
    board = Image.new("RGB", (max(wide.width, close.width), wide.height+4+close.height), "#262626")
    board.paste(wide, ((board.width-wide.width)//2, 0))
    board.paste(close, ((board.width-close.width)//2, wide.height+4))
    verdict = yield "frontage_verify", VERIFY, encoded(board)
    if verdict not in {"BAY", "APRON"}:
        return unresolved(NO_GROUND), "frontage_unverified"
    dx, dy = max(1, round(w*.023)), max(1, round(h*.016))
    return proposed([x,y], [max(0,x-dx),max(0,y-dy),min(w,x+dx),min(h,y+dy)]), "verified_landmark_frontage"
