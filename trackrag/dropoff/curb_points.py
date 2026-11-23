"""Describe observable unmarked parking areas using original pixels and the frozen vehicle tracks."""
from hashlib import sha256
from collections import OrderedDict
from copy import deepcopy
from io import BytesIO
from pathlib import Path
import re
from threading import Lock

from PIL import Image, ImageDraw

from . import bay_clearance, bay_landmarks, manual_surfaces, place_grounding
from .bay_packets import read_photo_packet
from .parking_evidence import _authenticate
from .passenger import digest


BANK = Path(__file__).resolve().parents[2] / "experiments/dropoff/passenger-curb-points-v1.zip"
RECIPE = "visible_unmarked_parking_points_v1"
CHECK = ("You are reviewing an original driver-eye photo. The small cyan ring marks one exact point. "
         "Classify the location of its centre. OPEN means visible unoccupied space beside the road, "
         "set back from the through traffic, where a vehicle could pull into the pictured area. "
         "LANE means roadway carrying through traffic or a junction, OCCUPIED means a vehicle or obstacle "
         "uses the space, and SEPARATED means a divider or footway prevents entry. If the point is on a "
         "thin strip next to a parked vehicle or you can't determine the area, answer UNCLEAR. "
         "Reply with one word: OPEN, LANE, OCCUPIED, SEPARATED or UNCLEAR.")
NEIGHBOURS = ("The cyan ring is on a patch of ground. Identify vehicles immediately flanking this patch "
              "on opposite ends of the SAME row, if visible. Give each vehicle's body colour and type "
              "(e.g. sedan, SUV, pickup, hatchback, flatbed truck or van). "
              "Use NONE for a missing neighbour. Two short phrases separated by |.")
FRONTAGE = ("Identify the visible storefront or building directly beside the cyan ring. Give its everyday "
            "category plus its printed name only if legible. Do not guess. No addresses, plates or people. "
            "Give a short phrase or NONE.")
ROADSIDE = ("What nearby physical landmark could a passenger use to describe the exact cyan-marked area? "
            "Look at its immediate surroundings for a divider, trees, gate, kerb, bin, sign, fence or gravel. "
            "Give at most two short phrases describing what is actually visible, or NONE.")
GUIDES = (CHECK, NEIGHBOURS, FRONTAGE, ROADSIDE)
WORDS = ("Classify the point inside the ring.", "Name the two vehicles beside this space.",
         "Read the frontage immediately beside this space.", "Describe the closest roadside landmarks.")


def source_context(scene):
    frame = scene["frame"]
    return {"size":[frame["width"], frame["height"]],
            "boxes":[t["bbox_xyxy"] for t in scene["tracks"] if t["frame_id"] == frame["frame_id"]]}


def locations(scene, classes):
    ctx = source_context(scene)
    road = bay_clearance.visible_road_component(classes)
    results = []
    for raw in place_grounding.surface_options(classes, ctx["size"], ctx["boxes"]):
        x0,y0,x1,y1 = raw["box"]
        option = {**raw, "polygon":[[x0,y0],[x1,y0],[x1,y1],[x0,y1]],
                  "evidence":{"recipe":RECIPE,"coverage":"visible_portion","frontage_xy":raw["point"],
                              "painted_boundaries":[],"boundary_widths":[],"boundary_opposite_closed":[]}}
        if road is not None and bay_clearance.on_visible_road(option,road,scene["frame"]):
            results.append(option)
    return results


def view(original, point, *, close):
    """Preserve context around the mark without hiding the ground being inspected."""
    w,h = original.size
    x,y = point
    if close:
        span = .66*w
        left = min(max(0, x-span/2),w-span)
        crop = (round(left), round(max(0,min(.20*h, y-.48*h))), round(left+span), h)
    else:
        crop = (0,0,w,h)
    im = original.crop(crop).convert("RGB")
    before = im.size
    im.thumbnail((1120,780),Image.Resampling.LANCZOS)
    cx,cy = (x-crop[0])*im.width/before[0], (y-crop[1])*im.height/before[1]
    draw = ImageDraw.Draw(im)
    for radius, color, width in ((8,"black",3),(6,"cyan",2)):
        draw.ellipse((cx-radius,cy-radius,cx+radius,cy+radius),outline=color,width=width)
    buffer = BytesIO()
    im.save(buffer,format="JPEG",quality=91,subsampling=0)
    return buffer.getvalue()


def get_packet(key):
    packet=read_photo_packet(BANK,key,60000)
    if packet is not None:
        return packet
    from . import local_8b_observations
    return local_8b_observations.get_packet(key,"points")


def observations(original, scene, options, packet):
    from . import local_8b_observations
    local=isinstance(packet,dict) and packet.get("origin")==local_8b_observations.ORIGIN
    if local:
        local_8b_observations.verify(packet,scene["provenance"]["image_sha256"],"points",scene=scene)
    else:
        _authenticate(packet)
    body = packet["body"]
    if (body.get("recipe")!=RECIPE or body.get("model")!=bay_landmarks.MODELS or
            body.get("guides")!=digest([GUIDES,WORDS]) or body.get("image_sha256")!=scene["provenance"]["image_sha256"] or
            body.get("source_context")!=digest(source_context(scene)) or original.size!=tuple(source_context(scene)["size"])):
        raise ValueError("The curb observations do not match this original and its current tracks.")
    rows = body.get("rows")
    if not isinstance(rows,list) or len(rows)!=len(options):
        raise ValueError("The curb observations must cover the full current candidate set.")
    result = []
    for option, row in zip(options, rows, strict=True):
        p = option["point"]
        replies = row.get("replies")
        if (row.get("point")!=p or row.get("views")!=[sha256(view(original,p,close=c)).hexdigest() for c in (False,True)]
                or not isinstance(replies,list) or len(replies)!=4 or
                any(not isinstance(reply,str) or len(reply)>240 for reply in replies)):
            raise ValueError("A verified curb point or its visual observations changed.")
        if replies[0].strip().upper().strip(".") != "OPEN":
            continue
        if local and (row.get("crop_verdict") or "").strip().upper().strip(".")!="OPEN":
            continue
        ends = replies[1].split("|")
        first,second = [part.strip() for part in ends] if len(ends)==2 else ("NONE","NONE")
        result.append({**option, "descriptions":{"nearer_vehicle": first, "farther_vehicle":second,
                                                "frontage":replies[2],"roadside":replies[3]}})
    return result


_cache = OrderedDict()
_lock = Lock()


def verified_options(scene, photo, surface, packet):
    key = scene["provenance"]["image_sha256"]
    if sha256(photo).hexdigest()!=key:
        raise ValueError("The original used for curb observations changed.")
    # The optical cache also serves public photos; rebind local-only identity before every cache hit.
    from . import local_8b_observations
    if isinstance(packet,dict) and packet.get("origin")==local_8b_observations.ORIGIN:
        local_8b_observations.verify(packet,key,"points",scene=scene)
    stamp = digest([key,source_context(scene),surface,packet])
    with _lock:
        if stamp in _cache:
            _cache.move_to_end(stamp)
            return deepcopy(_cache[stamp])
    classes=manual_surfaces.decode(surface,scene["frame"],key)
    options=locations(scene,classes)
    with Image.open(BytesIO(photo)) as original:
        options=observations(original,scene,options,packet)
    with _lock:
        _cache[stamp]=deepcopy(options)
        if len(_cache)>96:
            _cache.popitem(last=False)
    return options


def may_describe_point(words):
    # A point on unmarked pavement cannot establish a marked stall or permission to use it.
    return isinstance(words,str) and not re.search(r"\b(?:marked|painted|reserved|permit|bus stop|legal|illegal|safe)\b",words,re.I)
