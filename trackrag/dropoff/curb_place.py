"""Search the visible road next to a curb when an explicit parking surface did not resolve a place."""
from io import BytesIO
import re

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, distance_transform_edt

from .passenger import proposed, unresolved


LAYOUT = (
    "Look out of the passenger's car. Identify the particular landmark in the request. "
    "Classify the ground beside THAT landmark on the camera vehicle's accessible side. "
    "BAY = a visibly vacant recessed pull-in, layby or distinct parking bay outside the traffic lane. "
    "APRON = an open vehicle forecourt directly in front of an identifiable entrance, such as a porte-cochere. "
    "LANE = only the ordinary traffic lane or shoulder along a curb. DIVIDED = a raised median, barrier, "
    "fence or crossing separates the landmark from the vehicle-side ground. OCCUPIED = the area is full. "
    "UNCLEAR = the named place or distinct usable ground is missing or indistinguishable. "
    "A sign or building visible in the distance does not establish a pull-in. Reply with exactly one category."
)
CHOOSE = (
    "Find the landmark the passenger describes. Each numbered dot marks an actual point on road beside a curb. "
    "Which dot is INSIDE the corresponding empty pull-in or entrance forecourt on our side? "
    "Do not choose a passing lane, space in front of a parked bumper, raised surface, or a near-side point "
    "when the actual place is across a divider. If none is clearly inside the intended area, reply NONE. "
    "Otherwise reply with just its number."
)
VERIFY = (
    "The upper panel shows the full street. The lower panel magnifies the SAME yellow ring. "
    "Inspect where its exact centre lands relative to the passenger's particular landmark. "
    "Classify that point as BAY (inside the requested vacant, distinct off-lane pull-in), "
    "APRON (inside the requested open vehicle frontage immediately at the entrance), LANE (active lane or aisle), "
    "DIVIDED (wrong side of an obstacle), OCCUPIED, WRONG (different ground or landmark) or UNCLEAR. "
    "Visible curb paint, a shop sign or a ring on asphalt alone do not establish an empty pull-in. "
    "Return just one category."
)
LAYOUT_LABELS = {"BAY", "APRON", "LANE", "DIVIDED", "OCCUPIED", "UNCLEAR"}


def _frontage(classes, x, y, nearest):
    """Look beyond the curb. A median, hedge, parked row or second carriageway can separate a building."""
    h, w = classes.shape
    by, bx = nearest[:, y, x]
    dx, dy = float(bx-x), float(by-y)
    length = np.hypot(dx, dy)
    if length < 1:
        return False
    nx, ny = dx/length, dy/length
    offsets = np.linspace(-.4, .4, 5) * length
    steps = np.arange(-length*.45, min(h*.30, length*6 + h*.03), max(1.,h*.005))
    sx = np.rint(bx + nx*steps + ny*offsets[:,None]).astype(int)
    sy = np.rint(by + ny*steps - nx*offsets[:,None]).astype(int)
    votes = 0
    for ray_x, ray_y in zip(sx,sy):
        in_view = (ray_x>=0) & (ray_x<w) & (ray_y>=0) & (ray_y<h)
        line = classes[ray_y[in_view],ray_x[in_view]]
        edge = np.flatnonzero(np.isin(line,(2,15)))
        if not len(edge):
            continue
        # Trace past curb, footway and thin surface markings. The first sustained other surface matters.
        after = line[edge[0]+1:]
        if len(after)<2:
            continue
        solid = ~np.isin(after,(2,9,11,15,23,24,36))
        first = np.flatnonzero(solid[:-1] & (after[1:]==after[:-1]))
        if len(first) and after[first[0]] in (10,17):
            votes += 1
    return votes >= 3


def options(classes, size, boxes):
    """Sample supported ground on the road side of a detected curb or sidewalk, in source pixels."""
    height, width = classes.shape
    ground = np.isin(classes, (13, 14))  # Vistas Road and Service Lane
    # The boundary must touch drivable ground. A building edge or a parked car is not a curb.
    sidewalk = np.isin(classes, (2, 15))
    curb = ground & binary_dilation(sidewalk, iterations=2)
    if not curb.any():
        return []
    free = ground.copy()
    for left, top, right, bottom in boxes:
        margin = (right - left) * .08
        x0, x1 = np.array([left-margin, right+margin]) * width / size[0]
        y0, y1 = np.array([top, bottom+(bottom-top)*.3]) * height / size[1]
        free[max(0,int(np.floor(y0))):min(height,int(np.ceil(y1))),
             max(0,int(np.floor(x0))):min(width,int(np.ceil(x1)))] = False
    yy, xx = np.ogrid[:height, :width]
    depth = np.clip((yy / height - .34) / .66, 0, 1)
    # Allow more visible ground between the pin and the curb in the foreground.
    offset = height * (.012 + .062 * depth)
    edge_distance = distance_transform_edt(~curb)
    nearest = distance_transform_edt(~sidewalk, return_distances=False, return_indices=True)
    clearance = distance_transform_edt(np.pad(free, 1))[1:-1, 1:-1]
    supported = (free & (yy > height * .34) & (edge_distance >= offset * .45)
                 & (edge_distance <= offset * 1.65) & (clearance >= np.maximum(3., offset * .42)))
    score = np.where(supported, (1 - np.abs(edge_distance-offset)/offset)
                     * np.minimum(1., clearance/offset) * (.64 + .36 * depth), 0.)
    result = []
    for _ in range(60):
        cy, cx = np.unravel_index(score.argmax(), score.shape)
        if score[cy, cx] <= 0:
            break
        score[((xx-cx)/(width*.085))**2 + ((yy-cy)/(height*.084))**2 <= 1.] = 0.
        if not _frontage(classes,cx,cy,nearest):
            continue
        px, py = (cx + .5) * size[0]/width, (cy + .5) * size[1]/height
        radius = min(float(clearance[cy,cx]) * .40, width * .010)
        dx, dy = radius * size[0]/width, radius * size[1]/height
        result.append({"point":[float(px),float(py)], "box":[float(px-dx),float(py-dy),float(px+dx),float(py+dy)]})
        if len(result)==12:
            break
    return sorted(result, key=lambda item: tuple(item["point"]))


def inspection_view(original, choices, selected):
    from .place_grounding import frame_view
    wide = frame_view(original, choices, selected)
    wide.thumbnail((780, 430), Image.Resampling.LANCZOS)
    point = choices[selected-1]["point"]
    w,h = original.size
    cw,ch = max(1,round(w*.50)),max(1,round(h*.48))
    left = max(0,min(w-cw,round(point[0]-cw/2)))
    top = max(0,min(h-ch,round(point[1]-ch*.70)))
    close = original.crop((left,top,left+cw,top+ch)).convert("RGB")
    close.thumbnail((780, 380), Image.Resampling.LANCZOS)
    x,y = (point[0]-left)*close.width/cw, (point[1]-top)*close.height/ch
    draw = ImageDraw.Draw(close)
    for radius, colour in ((9,"black"),(7,"#ffdc33")):
        draw.ellipse((x-radius,y-radius,x+radius,y+radius),outline=colour,width=2)
    board = Image.new("RGB",(max(wide.width,close.width),wide.height+4+close.height),"#262626")
    board.paste(wide,((board.width-wide.width)//2,0))
    board.paste(close,((board.width-close.width)//2,wide.height+4))
    out = BytesIO()
    board.save(out,format="JPEG",quality=92,subsampling=0)
    return out.getvalue()


def steps(context, classes, original):
    from .place_grounding import QUESTION, view_bytes
    choices = options(classes, context["size"], context["current_boxes"])
    if not choices:
        return unresolved(QUESTION), "no_visible_curb_ground"
    layout = yield "curb_layout", LAYOUT, view_bytes(original, [])
    if layout not in LAYOUT_LABELS:
        raise ValueError("Invalid curb layout response")
    if layout not in {"BAY","APRON"}:
        return unresolved(QUESTION), "curb_layout_" + layout.lower()
    raw = yield "curb_choose", CHOOSE, view_bytes(original, choices)
    if raw == "NONE":
        return unresolved(QUESTION), "no_matching_curb_place"
    if not isinstance(raw,str) or not re.fullmatch(r"[1-9]\d?",raw) or not 1 <= int(raw) <= len(choices):
        raise ValueError("Invalid curb candidate response")
    chosen = int(raw)
    verdict = yield "curb_check", VERIFY, inspection_view(original, choices, chosen)
    if verdict not in LAYOUT_LABELS | {"WRONG"}:
        raise ValueError("Invalid curb point assessment")
    if verdict not in {"BAY","APRON"}:
        return unresolved(QUESTION), "curb_point_" + verdict.lower()
    item = choices[chosen-1]
    return proposed(item["point"],item["box"]), "visible_curb_" + verdict.lower()
